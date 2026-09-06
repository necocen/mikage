# Temporary WASM compatibility patch

Source: the unmodified `egui-winit` 0.36.1 release archive from
https://static.crates.io/crates/egui-winit/egui-winit-0.36.1.crate.
Upstream: https://github.com/emilk/egui/issues/8436.

The only Rust change is `src/dropped_file.rs`: `NativeFile::bytes` is native-only;
on wasm32 it implements the required `bytes_async` method. A native path has no
browser File handle, so that method returns an explicit unsupported-path error.
This does not add browser file-drop support or change native file reading.

Mikage depends directly on this directory so git/path consumers get the fix too;
a root-only `[patch.crates-io]` would not propagate to dependent applications.
The upstream MIT and Apache-2.0 licenses are included here. The normalized package
manifest's license include paths have been adjusted to this directory.

Remove this directory and switch both target dependencies back to crates.io once
an upstream release implements the WASM DroppedFile contract, then rerun the native
and WASM feature matrix. Publishing a registry release of mikage requires replacing
this path dependency with an upstream fixed release or a published patched crate;
do not silently substitute the broken unpatched 0.36.1 package.

The Rust delta against the release archive is:

```diff
 impl egui::DroppedFile for NativeFile {
     // path() is unchanged.
+    #[cfg(not(target_arch = "wasm32"))]
     fn bytes(&self) -> Result<Vec<u8>, String> {
         std::fs::read(&self.path).map_err(|err| err.to_string())
     }
+
+    #[cfg(target_arch = "wasm32")]
+    fn bytes_async(
+        &self,
+    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<u8>, String>> + '_>> {
+        // winit supplies a native path, not a browser File handle. Browsers cannot
+        // read such paths; report this explicitly instead of attempting std::fs.
+        Box::pin(async { Err("native file paths cannot be read in a browser".to_owned()) })
+    }
 }
```
