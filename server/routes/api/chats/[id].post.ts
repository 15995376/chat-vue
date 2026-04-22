import type { UIMessage } from 'ai'
import { createUIMessageStream, createUIMessageStreamResponse } from 'ai'
import { z } from 'zod'
import { useUserSession } from '../../../utils/session'
import { useDrizzle, tables, eq, and } from '../../../utils/drizzle'
import { defineHandler, HTTPError } from 'nitro'
import { getValidatedRouterParams, readValidatedBody } from 'nitro/h3'
import { MODELS } from '../../../../shared/utils/models'

// FastGPT API 配置
const FASTGPT_API_URL = process.env.FASTGPT_API_URL || 'https://api.fastgpt.in'
const FASTGPT_API_KEY = process.env.FASTGPT_API_KEY || ''

// 将 UIMessage 转换为 FastGPT 消息格式
function convertToFastGPTMessages(messages: UIMessage[]) {
  return messages.map(msg => {
    // 提取文本内容
    let content = ''
    if (msg.parts) {
      for (const part of msg.parts) {
        if (part.type === 'text') {
          content += part.text
        }
      }
    }
    return {
      role: msg.role as 'user' | 'assistant' | 'system',
      content
    }
  })
}

// 生成标题的函数
async function generateTitleFromFastGPT(message: UIMessage): Promise<string> {
  let content = ''
  if (message.parts) {
    for (const part of message.parts) {
      if (part.type === 'text') {
        content += part.text
      }
    }
  }

  try {
    const response = await fetch(`${FASTGPT_API_URL}/api/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${FASTGPT_API_KEY}`
      },
      body: JSON.stringify({
        stream: false,
        messages: [
          {
            role: 'system',
            content: `You are a title generator for a chat:
          - Generate a short title based on the first user's message
          - The title should be less than 30 characters long
          - The title should be a summary of the user's message
          - Do not use quotes (' or ") or colons (:) or any other punctuation
          - Do not use markdown, just plain text`
          },
          {
            role: 'user',
            content
          }
        ]
      })
    })

    if (!response.ok) {
      console.error('[FastGPT] Title generation failed:', response.statusText)
      return content.slice(0, 30)
    }

    const data = await response.json() as { choices?: { message?: { content?: string } }[] }
    return data.choices?.[0]?.message?.content?.slice(0, 30) || content.slice(0, 30)
  } catch (error) {
    console.error('[FastGPT] Title generation error:', error)
    return content.slice(0, 30)
  }
}

export default defineHandler(async (event) => {
  const session = await useUserSession(event)

  const { id } = await getValidatedRouterParams(event, z.object({
    id: z.string()
  }).parse)

  const { messages } = await readValidatedBody(event, z.object({
    model: z.string().refine(value => MODELS.some(m => m.value === value), {
      message: 'Invalid model'
    }),
    messages: z.array(z.custom<UIMessage>())
  }).parse)

  const db = useDrizzle()

  const chat = await db.query.chats.findFirst({
    where: (chat, { eq }) => and(eq(chat.id, id as string), eq(chat.userId, session.data.user?.id || session.id!)),
    with: {
      messages: true
    }
  })
  if (!chat) {
    throw new HTTPError({ statusCode: 404, statusMessage: 'Chat not found' })
  }

  if (!chat.title) {
    const title = await generateTitleFromFastGPT(messages[0])
    await db.update(tables.chats).set({ title }).where(eq(tables.chats.id, id as string))
  }

  const lastMessage = messages[messages.length - 1]
  if (lastMessage?.role === 'user' && messages.length > 1) {
    await db.insert(tables.messages).values({
      id: lastMessage.id,
      chatId: id as string,
      role: 'user',
      parts: lastMessage.parts
    }).onConflictDoUpdate({ target: tables.messages.id, set: { parts: lastMessage.parts } })
  }

  const abortController = new AbortController()
  event.runtime?.node?.req?.on('close', () => abortController.abort())

  const stream = createUIMessageStream({
    execute: async ({ writer }) => {
      // 调用 FastGPT API
      const fastgptMessages = convertToFastGPTMessages(messages)

      const response = await fetch(`${FASTGPT_API_URL}/api/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${FASTGPT_API_KEY}`
        },
        body: JSON.stringify({
          stream: true,
          messages: fastgptMessages
        }),
        signal: abortController.signal
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`FastGPT API error: ${response.status} ${errorText}`)
      }

      if (!chat.title) {
        writer.write({
          type: 'data-chat-title',
          data: { message: 'Generating title...' },
          transient: true
        })
      }

      // 处理 SSE 流
      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('No response body')
      }

      const decoder = new TextDecoder()
      let buffer = ''

      // 生成消息 ID
      const messageId = crypto.randomUUID()

      // 开始新消息
      writer.write({
        type: 'start',
        id: messageId,
        role: 'assistant'
      })

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed || !trimmed.startsWith('data:')) continue

          const data = trimmed.slice(5).trim()
          if (data === '[DONE]') continue

          try {
            const parsed = JSON.parse(data) as {
              choices?: {
                delta?: { content?: string }
              }[]
            }
            const content = parsed.choices?.[0]?.delta?.content
            if (content) {
              writer.write({
                type: 'text',
                text: content
              })
            }
          } catch {
            // 忽略解析错误
          }
        }
      }

      // 结束消息
      writer.write({
        type: 'finish',
        finishReason: 'stop'
      })
    },
    onFinish: async ({ messages }) => {
      await db.insert(tables.messages).values(messages.map(message => ({
        id: message.id,
        chatId: chat.id,
        role: message.role as 'user' | 'assistant',
        parts: message.parts
      }))).onConflictDoNothing()
    }
  })

  return createUIMessageStreamResponse({
    stream
  })
})
