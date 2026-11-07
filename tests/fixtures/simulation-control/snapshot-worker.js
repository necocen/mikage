// A CPU-only Worker publishes data; only the window host calls the GPU APIs.
self.onmessage = ({ data }) => postMessage({ value: data.value });
