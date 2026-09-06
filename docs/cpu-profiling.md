# Native CPU profiling

For a wasm deployment using wasi-nn, set `nnCpuProfile: true` in its
configuration and restart it. The native gperftools library is loaded only
for that process. Sampling remains off until the owner requests a capture.
This is diagnostic instrumentation; measure normal throughput with the
option disabled.

Using the deployment owner's bearer token against the node supervisor:

* `POST /v1/deployments/:id/cpu-profile` with
  `{"action":"start","seconds":30}` starts one capture, at 49 Hz.
  Durations must be integers from 1 through 60 seconds.
* The capture stops automatically. `{"action":"stop"}` stops it early.
* `GET /v1/deployments/:id/cpu-profile` returns its state. Once `complete`,
  `data` contains the base64-encoded gperftools profile, up to 16 MiB.
  Decode it to a private file and inspect it with Google's `pprof`, using
  the executable and native libraries from the exact deployed image.

There is one capture per process generation. Repeated start/stop requests
are idempotent; restart the deployment for another capture. A process exit
or failed stop acknowledgement marks the profile incomplete rather than
presenting a potentially truncated capture as valid.

Profiles contain instruction addresses and process mappings. They are
owner-only even for public deployments, stored outside WASI preopens, and
removed when that process generation is torn down. Neither the output
path nor the preloaded library can be supplied by the app.

Sampling measures CPU execution, including background worker threads. It
does not measure time blocked waiting for GPU work. Interpret CPU samples
alongside the engine's request/phase timings; do not add background CPU
time to request wall time.
