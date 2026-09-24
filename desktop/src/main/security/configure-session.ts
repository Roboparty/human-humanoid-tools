import type { Session } from 'electron'

export function configureDesktopSession(session: Session, origin: string, secret: string): void {
  // The current desktop feature set needs no camera, microphone, geolocation, or notifications.
  session.setPermissionCheckHandler(() => false)
  session.setPermissionRequestHandler((_webContents, _permission, callback) => callback(false))

  // Inject authentication in Electron's network layer so renderer JavaScript cannot read the secret.
  session.webRequest.onBeforeSendHeaders({ urls: [`${origin}/*`] }, (details, callback) => {
    details.requestHeaders['X-HHTools-Session'] = secret
    callback({ requestHeaders: details.requestHeaders })
  })
}
