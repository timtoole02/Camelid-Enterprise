/* Assistant avatar — frames the Camelid mark (
   the derivative four-point glyph is fully retired). */
import CamelidMark from './CamelidMark'

export function Avatar({ size = 30, className = '', state = 'idle', pulse = 0 }) {
  return (
    <span className={`camelid-avatar ${className}`.trim()} style={{ '--avatar-size': `${size}px` }} aria-hidden="true">
      <CamelidMark size={Math.round(size * 0.72)} state={state} pulse={pulse} />
    </span>
  )
}

export default Avatar
