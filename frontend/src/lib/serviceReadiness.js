/* When may this console send a prompt?
 *
 * This replaces a readiness gate carried over from the single-user application,
 * which asked a second question this product has no way to answer: whether the
 * selected model matched an exact row in a published compatibility contract. A
 * replica serves ten contractual routes and none of them serves that contract —
 * so every model would have failed a check whose evidence can never arrive, and
 * the console would have sat permanently blocked while the service behind it was
 * serving fine.
 *
 * The replacement asks only what a replica actually publishes on `/v1/health`:
 *
 *   - `loaded_now`        — weights are resident
 *   - `generation_ready`  — it can decode
 *   - `active_model_id`   — which model that is
 *
 * A replica serves exactly one model for its whole process lifetime — the file
 * it hashed before it bound its port — so "is my selected model the served one"
 * is a comparison against `active_model_id`, not a request to go load something.
 * There is no load/unload here because there is no route for it, and there is no
 * route for it because a model swap over the serving port would invalidate the
 * identity digests the replica publishes on every response.
 */

export const Readiness = {
  /* The gateway or replica did not answer at all. */
  Offline: 'offline',
  /* Answering, but no weights resident yet. */
  NoModel: 'no-model',
  /* Weights resident, not yet able to decode — the window during warmup. */
  Loading: 'loading',
  /* Serving a model, but not the one selected in the UI. */
  OtherModel: 'other-model',
  Ready: 'ready',
}

export function getServiceReadiness(runtime, selectedModelId = null) {
  if (!runtime || runtime.status === 'offline') {
    return {
      state: Readiness.Offline,
      canSend: false,
      label: 'Service unavailable',
      copy: 'The gateway did not respond. Check the endpoint in Settings; drafts stay editable meanwhile.',
    }
  }

  if (!runtime.loaded_now) {
    return {
      state: Readiness.NoModel,
      canSend: false,
      label: 'No model loaded',
      copy: 'This replica has not reported a resident model. A replica loads its model at startup.',
    }
  }

  if (!runtime.generation_ready) {
    return {
      state: Readiness.Loading,
      canSend: false,
      label: 'Preparing',
      copy: 'The model is resident but not yet able to decode. This clears on its own.',
    }
  }

  const served = runtime.active_model_id || null

  /* No selection yet is not a fault — the console picks the served model by
     default, so this is only reachable before the first health response. */
  if (selectedModelId && served && selectedModelId !== served) {
    return {
      state: Readiness.OtherModel,
      canSend: false,
      served,
      label: 'Different model served',
      copy: `This endpoint serves ${served}. A replica serves one model for its process lifetime; point the console at the endpoint serving ${selectedModelId}.`,
    }
  }

  return {
    state: Readiness.Ready,
    canSend: true,
    served,
    label: 'Ready',
    copy: served ? `Serving ${served}.` : 'Ready to send.',
  }
}

/* Which of the listed models this endpoint can actually answer for: the served
   one, and only while it is generation-ready. */
export function isModelServable(model, runtime) {
  if (!model || !runtime || !runtime.generation_ready) return false
  const id = model.id || model.name
  return Boolean(id && runtime.active_model_id && id === runtime.active_model_id)
}
