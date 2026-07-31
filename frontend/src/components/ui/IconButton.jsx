/* IconButton — square, accessible, label required. variants: ghost | tonal | solid */
export function IconButton({
  label,
  variant = 'ghost',
  size = 'md',
  active = false,
  className = '',
  children,
  ...rest
}) {
  const classes = [
    'cx-iconbtn',
    `cx-iconbtn--${variant}`,
    `cx-iconbtn--${size}`,
    active ? 'is-active' : '',
    className,
  ].filter(Boolean).join(' ')

  return (
    <button type="button" className={classes} aria-label={label} title={label} aria-pressed={active || undefined} {...rest}>
      {children}
    </button>
  )
}

export default IconButton
