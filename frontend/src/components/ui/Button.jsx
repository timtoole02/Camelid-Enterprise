/* Button — variants: primary | tonal | ghost | danger | outline; sizes: sm | md | lg */
export function Button({
  variant = 'tonal',
  size = 'md',
  icon = null,
  iconRight = null,
  block = false,
  loading = false,
  className = '',
  children,
  type = 'button',
  ...rest
}) {
  const classes = [
    'cx-btn',
    `cx-btn--${variant}`,
    `cx-btn--${size}`,
    block ? 'cx-btn--block' : '',
    loading ? 'is-loading' : '',
    !children ? 'cx-btn--icon-only' : '',
    className,
  ].filter(Boolean).join(' ')

  return (
    <button type={type} className={classes} aria-busy={loading || undefined} {...rest}>
      {loading && <span className="cx-btn__spinner" aria-hidden="true" />}
      {!loading && icon && <span className="cx-btn__icon">{icon}</span>}
      {children && <span className="cx-btn__label">{children}</span>}
      {!loading && iconRight && <span className="cx-btn__icon">{iconRight}</span>}
    </button>
  )
}

export default Button
