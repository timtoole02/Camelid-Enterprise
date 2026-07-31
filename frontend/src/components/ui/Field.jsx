/* Field — labeled form control wrapper. Pass an <input>/<textarea>/<select> or children. */
let fieldSeq = 0

export function Field({ label, hint, htmlFor, className = '', children }) {
  const id = htmlFor || `cx-field-${(fieldSeq += 1)}`
  return (
    <label className={`cx-field ${className}`.trim()} htmlFor={id}>
      {label && <span className="cx-field__label">{label}</span>}
      {typeof children === 'function' ? children(id) : children}
      {hint && <span className="cx-field__hint">{hint}</span>}
    </label>
  )
}

export default Field
