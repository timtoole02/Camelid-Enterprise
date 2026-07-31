import { Button } from './Button'
import { Modal } from './Modal'

/* ConfirmDialog — destructive/confirm prompt. Replaces ConversationDeleteDialog. */
export function ConfirmDialog({
  open,
  title = 'Are you sure?',
  detail = '',
  confirmLabel = 'Delete',
  cancelLabel = 'Cancel',
  tone = 'danger',
  busy = false,
  onCancel,
  onConfirm,
}) {
  return (
    <Modal
      open={open}
      onClose={busy ? () => {} : onCancel}
      title={title}
      labelledById="cx-confirm-title"
      size="sm"
      footer={
        <>
          <Button variant="ghost" onClick={onCancel} disabled={busy}>{cancelLabel}</Button>
          <Button variant={tone} onClick={onConfirm} loading={busy}>{confirmLabel}</Button>
        </>
      }
    >
      {detail && <p className="cx-confirm__detail">{detail}</p>}
    </Modal>
  )
}

export default ConfirmDialog
