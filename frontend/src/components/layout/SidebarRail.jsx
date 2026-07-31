import { useMemo } from 'react'
import { CamelidMark } from '../ui/CamelidMark'
import { StatusDot } from '../ui/StatusDot'
import { ThemeToggle } from '../ui/ThemeToggle'
import { Tooltip } from '../ui/Tooltip'
import { ConversationListItem } from './ConversationListItem'
import {
  IconAnalytics, IconApi, IconBolt, IconChart, IconChat, IconHistory, IconMemory, IconModels,
  IconDownload, IconNetwork, IconNewChat, IconObservatory, IconReceipt, IconSearch, IconSettings, IconSidebar, IconSystem,
} from '../ui/icons'

/* Every entry here is a view the console actually renders. The list this
   replaces carried twelve, most of them surfaces of the single-user
   application — model management, downloads, cluster topology — whose data came
   from routes a replica does not serve. A nav item that leads nowhere is worse
   than an absent one: it reads as a feature that is broken rather than one that
   was never part of this product. */
const NAV = [
  { tab: 'history', label: 'Chat history', Icon: IconHistory },
  { tab: 'settings', label: 'Settings', Icon: IconSettings },
]

const BUCKETS = ['Today', 'Yesterday', 'Previous 7 days', 'Earlier']
function startOfDay(d) { return new Date(d.getFullYear(), d.getMonth(), d.getDate()) }
function bucketFor(value) {
  if (!value) return 'Earlier'
  const diff = Math.floor((startOfDay(new Date()) - startOfDay(new Date(value))) / 86400000)
  if (diff <= 0) return 'Today'
  if (diff === 1) return 'Yesterday'
  if (diff <= 7) return 'Previous 7 days'
  return 'Earlier'
}

export function SidebarRail({
  collapsed,
  onToggleCollapsed,
  showNewChatLanding,
  search,
  setSearch,
  tab,
  setTab,
  filteredConversations,
  selectedConversationId,
  onSelectConversation,
  renameConversation,
  requestDeleteConversation,
  runtime,
  themePreference,
  themeResolved,
  onCycleTheme,
}) {
  const grouped = useMemo(() => {
    const groups = new Map(BUCKETS.map((b) => [b, []]))
    filteredConversations.forEach((c) => groups.get(bucketFor(c.updated_at))?.push(c))
    return BUCKETS.map((label) => ({ label, items: groups.get(label) || [] })).filter((g) => g.items.length)
  }, [filteredConversations])

  const online = runtime?.status === 'online'
  const statusTone = online ? 'ready' : 'offline'
  const statusLabel = online ? 'Service online' : 'Service offline'

  if (collapsed) {
    return (
      <aside className="rail rail--collapsed" id="camelid-sidebar" aria-label="Navigation rail">
        <div className="rail__rail-top">
          <Tooltip content="Expand sidebar" placement="right">
            <button type="button" className="rail__icon-btn" aria-label="Expand sidebar" onClick={onToggleCollapsed}>
              <IconSidebar size={20} />
            </button>
          </Tooltip>
          <Tooltip content="New chat" placement="right">
            <button type="button" className="rail__icon-btn rail__icon-btn--accent" aria-label="New chat" onClick={showNewChatLanding}>
              <IconNewChat size={20} />
            </button>
          </Tooltip>
        </div>
        <nav className="rail__rail-nav" aria-label="Primary">
          <Tooltip content="Chat" placement="right">
            <button type="button" className={`rail__icon-btn ${tab === 'chat' ? 'is-active' : ''}`} aria-label="Chat" aria-current={tab === 'chat' ? 'page' : undefined} onClick={() => setTab('chat')}>
              <IconChat size={20} />
            </button>
          </Tooltip>
          {NAV.map(({ tab: t, label, Icon }) => (
            <Tooltip key={t} content={label} placement="right">
              <button type="button" className={`rail__icon-btn ${tab === t ? 'is-active' : ''}`} aria-label={label} aria-current={tab === t ? 'page' : undefined} onClick={() => setTab(t)}>
                <Icon size={20} />
              </button>
            </Tooltip>
          ))}
        </nav>
        <div className="rail__rail-bottom">
          <ThemeToggle preference={themePreference} resolved={themeResolved} onCycle={onCycleTheme} compact />
          <Tooltip content={statusLabel} placement="right">
            <span className="rail__status-icon"><StatusDot tone={statusTone} pulse={online} /></span>
          </Tooltip>
        </div>
      </aside>
    )
  }

  return (
    <aside className="rail" id="camelid-sidebar" aria-label="Navigation sidebar">
      <div className="rail__header">
        <button type="button" className="rail__brand" onClick={showNewChatLanding} aria-label="Camelid home">
          <CamelidMark size={24} />
          <span className="rail__brand-name">Camelid</span>
        </button>
        <button type="button" className="rail__icon-btn" aria-label="Collapse sidebar" onClick={onToggleCollapsed}>
          <IconSidebar size={20} />
        </button>
      </div>

      <button type="button" className="rail__new-chat" onClick={showNewChatLanding}>
        <IconNewChat size={18} />
        <span>New chat</span>
      </button>

      <div className="rail__search">
        <IconSearch size={16} />
        <input
          className="rail__search-input"
          aria-label="Search conversations"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search chats"
        />
      </div>

      <div className="rail__scroll">
        <div className="rail__section">
          <div className="rail__section-label">Recent</div>
          {grouped.length === 0 && <p className="rail__empty">No conversations yet</p>}
          {grouped.map((group) => (
            <div key={group.label} className="rail__group">
              <div className="rail__group-label">{group.label}</div>
              {group.items.map((conversation) => (
                <ConversationListItem
                  key={conversation.id}
                  conversation={conversation}
                  collapsed={false}
                  selected={tab === 'chat' && conversation.id === selectedConversationId}
                  onSelect={onSelectConversation}
                  onRename={renameConversation}
                  onDelete={requestDeleteConversation}
                />
              ))}
            </div>
          ))}
        </div>

        <nav className="rail__section rail__nav" aria-label="Sections">
          <div className="rail__section-label">Sections</div>
          {NAV.map(({ tab: t, label, Icon }) => (
            <button
              key={t}
              type="button"
              className={`rail__nav-item ${tab === t ? 'is-active' : ''}`}
              aria-current={tab === t ? 'page' : undefined}
              onClick={() => setTab(t)}
            >
              <Icon size={20} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
      </div>

      <div className="rail__footer">
        <ThemeToggle preference={themePreference} resolved={themeResolved} onCycle={onCycleTheme} />
        <span className="rail__status"><StatusDot tone={statusTone} pulse={online} label={statusLabel} /></span>
      </div>
    </aside>
  )
}

export default SidebarRail
