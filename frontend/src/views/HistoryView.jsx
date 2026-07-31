import { clampText, formatCompactNumber, formatHistoryDate, formatPreview } from '../lib/formatters'
import { Button } from '../components/ui/Button'
import { downloadConversation } from '../lib/conversationExport'
import { EmptyState } from '../components/ui/EmptyState'
import { IconHistory, IconChat, IconTrash } from '../components/ui/icons'

function getConversationStats(conversation) {
  const messageCount = conversation.messages?.length || 0
  const assistantCount = conversation.messages?.filter((message) => message.role === 'assistant').length || 0
  const latestMessage = conversation.messages?.[messageCount - 1]
  return { messageCount, assistantCount, latestMessage }
}

export default function HistoryView({ filteredConversations, setSelectedConversationId, setTab, deleteConversation }) {
  const totalMessages = filteredConversations.reduce((sum, c) => sum + (c.messages?.length || 0), 0)
  const activeToday = filteredConversations.filter((c) => {
    if (!c.updated_at) return false
    return new Date(c.updated_at).toDateString() === new Date().toDateString()
  }).length
  const hasConversations = filteredConversations.length > 0

  return (
    <section className="history-view cxv">
      <header className="cxv-head">
        <div className="cxv-head__copy">
          <p className="cxv-kicker"><IconHistory size={14} /> Chat history</p>
          <h1>Conversations</h1>
          <p className="cxv-sub">Every thread stays searchable on this machine — jump back into earlier work, skim what changed, and pick up the next prompt without losing context.</p>
        </div>
        <div className="cxv-stats">
          <div className="cxv-stat">
            <span>Conversations</span>
            <strong>{formatCompactNumber(filteredConversations.length)}</strong>
            <small>{hasConversations ? `${activeToday} updated today` : 'None match this view'}</small>
          </div>
          <div className="cxv-stat">
            <span>Messages</span>
            <strong>{formatCompactNumber(totalMessages)}</strong>
            <small>stored locally</small>
          </div>
        </div>
      </header>

      {!hasConversations ? (
        <EmptyState
          icon={<IconHistory size={26} />}
          title="No conversations yet"
          description="No threads matched this view. Try a broader search, or start a fresh chat to build local history."
          action={<Button variant="primary" icon={<IconChat size={16} />} onClick={() => setTab('chat')}>New chat</Button>}
        />
      ) : (
        <div className="cxv-grid">
          {filteredConversations.map((conversation) => {
            const { messageCount, assistantCount, latestMessage } = getConversationStats(conversation)
            return (
              <article key={conversation.id} className="cxv-card">
                <header className="cxv-card__head">
                  <div className="cxv-card__titles">
                    <strong title={conversation.title || 'Untitled chat'}>{clampText(conversation.title || 'Untitled chat', 70) || 'Untitled chat'}</strong>
                    <span className="cxv-card__sub" title={conversation.model_id || 'No model recorded'}>{conversation.model_id || 'No model recorded'}</span>
                  </div>
                  <span className="cxv-tag">{formatCompactNumber(messageCount)} msgs</span>
                </header>

                <p className="cxv-card__preview">{formatPreview(latestMessage?.content, 180)}</p>

                <footer className="cxv-card__foot">
                  <div className="cxv-card__meta">
                    <span><strong>{formatCompactNumber(assistantCount)}</strong> replies</span>
                    <span className="cxv-dot">·</span>
                    <span title={conversation.updated_at || 'Unknown'}>{formatHistoryDate(conversation.updated_at) || 'Unknown'}</span>
                  </div>
                  <div className="cxv-card__actions">
                    <Button
                      variant="tonal"
                      size="sm"
                      icon={<IconChat size={15} />}
                      onClick={() => { setSelectedConversationId(conversation.id); setTab('chat') }}
                    >
                      Open
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      title="Export as Markdown (local download; never includes file paths)"
                      onClick={() => downloadConversation(conversation, 'markdown')}
                    >
                      MD
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      title="Export as JSON (local download; never includes file paths)"
                      onClick={() => downloadConversation(conversation, 'json')}
                    >
                      JSON
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="cxv-danger"
                      icon={<IconTrash size={15} />}
                      aria-label="Delete conversation"
                      onClick={() => deleteConversation(conversation.id)}
                    />
                  </div>
                </footer>
              </article>
            )
          })}
        </div>
      )}
    </section>
  )
}
