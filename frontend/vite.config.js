import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

/* Deliberately minimal.
 *
 * The configuration this replaces carried a dev-server middleware that could
 * spawn and kill a local inference process over `/__camelid/backend/launch`.
 * That is reasonable for a desktop application shipping its own engine; it is
 * the wrong posture for a console whose whole premise is that the service is
 * deployed and operated somewhere else, by someone else.
 *
 * There is no dev proxy either. The console talks to a gateway origin the user
 * sets at sign-in and the gateway applies a permissive CORS policy, so a proxy
 * would only paper over a misconfiguration in development that would still be
 * there in production.
 */
export default defineConfig({
  plugins: [react()],
  build: {
    // The console is served as static assets by whatever fronts it; nothing
    // here assumes a particular mount path.
    assetsDir: 'assets',
  },
})
