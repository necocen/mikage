# Changelog

## v0.3.0

### Added

- **MSAA support** — `RunConfig::sample_count` で 4x MSAA を有効化可能。`GpuContext::sample_count()` / `msaa_view()` で現在の設定を取得できる。`sample_count` に不正値を渡すと起動時に panic する（1 または 4 のみ許可）。
- **`RenderTargetConfig`** — `GpuContext::render_target_config()` でカラーフォーマット・深度フォーマット・サンプル数を一括取得。`color_target_state()` / `multisample_state()` / `depth_stencil_state()` でパイプライン作成のボイラープレートを削減。
- **`FrameContext::color_attachment()`** — MSAA resolve を自動処理するカラーアタッチメントビルダー。
- **`InstanceRenderer::prepare_compute()`** — compute shader 統合用のワンコール API。バッファ確保・カウント設定・再割当て検出をまとめて行う。`ComputeBufferState` を返す。
- **`InteractiveCamera` ジェスチャーフック** — `on_touch_drag()` / `on_touch_drag_end()` / `on_pinch_pan()` をトレイトメソッドとして追加。デフォルト実装はマウス/スクロール系メソッドに委譲。カスタムカメラでジェスチャーごとの挙動をオーバーライド可能に。
- **`InteractiveCamera::set_viewport_size()` / `set_cursor_position()`** — ウィンドウサイズ・カーソル位置をカメラに通知するトレイトメソッド。
- **高レベル API** — `SolidRenderer::add_object()` / `update_object()` / `InstanceRenderer::update_instances()` が `&GpuContext` を受け取るようになり、raw wgpu 型を直接扱う必要がなくなった。旧 API は `*_raw()` として残存（`#[doc(hidden)]`、安定 API 外）。

### Changed

- **`FrameContext::surface_view` が非公開に** — `ctx.surface_view` の直接参照を `ctx.color_attachment(ops)` に置き換える必要がある。MSAA 有無を問わず正しく動作する。
- **`InteractiveCamera::enabled` が非公開に** — `set_enabled()` / `is_enabled()` トレイトメソッドでアクセス。`enabled` は入力イベントの抑制のみを制御し、慣性運動（`update()` 内）は継続する。
- **Camera2d パン精度向上** — ピクセル→ワールド変換ベースに変更。ハードコードされていた `pan_speed` フィールドを削除。ズーム時はカーソル下のワールド座標が保存される（zoom-to-cursor）。
- **フレームレート非依存ダンピング** — `OrbitCamera` / `Camera2d` の慣性減衰が正確な等比級数で計算されるようになり、フレームレートによる挙動のばらつきを解消。
- **ドラッグライフサイクル修正** — ボタン状態のゲーティング、全ボタン解放時の `on_drag_end`、egui キャプチャ遷移時の正しいドラッグ終了処理。
- **egui イベントフィルタリング改善** — キーボード・ポインタイベントがカテゴリ別に正確にフィルタリングされるようになり、egui 操作中のスタック入力を防止。
- **`SolidObjectId` の `Default` を削除** — 無効な ID でパニックする罠を排除。`update_object()` で不正 ID は明示的な `expect` メッセージ付きでパニック。
- **`InputState::begin_frame` → `end_frame`** に改名、`pub(crate)` に変更（内部 API）。
- **SurfaceError 復帰改善** — `render_frame()` の全エラーアームで `request_redraw()` を呼び出し、WASM でレンダーループが無音停止する問題を修正。
- **`SceneUniform::with_light` NaN 防止** — ゼロベクトルの `light_dir` が `Vec3::Y` にフォールバックするように変更。

### Documentation

- `SolidRenderer` の透明オブジェクトポリシーを文書化（alpha による自動パイプライン切替、depth sort なしの制約）。
- `#[doc(hidden)]` メソッド群に「integration test 用、安定 API 外」の注記を追加。
- テスト内の `unsafe mem::zeroed()` KeyEvent ヘルパーに SAFETY コメントと TODO を追加。
- `InteractiveCamera` のジェスチャーデフォルト仮定を文書化。

### Breaking changes migration guide

```rust
// FrameContext: surface_view → color_attachment()
// Before (v0.2):
let attachment = wgpu::RenderPassColorAttachment {
    view: ctx.surface_view,
    resolve_target: None,
    ops: ...,
};
// After (v0.3):
let attachment = ctx.color_attachment(ops);

// InteractiveCamera: enabled フィールド → メソッド
// Before (v0.2):
camera.enabled = false;
if camera.enabled { ... }
// After (v0.3):
camera.set_enabled(false);
if camera.is_enabled() { ... }

// SolidObjectId: Default 削除
// Before (v0.2):
let id = SolidObjectId::default(); // コンパイルできるが危険
// After (v0.3):
let id = solid.add_object(&gpu, &positions, &normals, &indices);

// Camera2d: pan_speed 削除
// Before (v0.2):
let cam = Camera2d { pan_speed: 0.01, ..Default::default() };
// After (v0.3):
let cam = Camera2d::default(); // パン速度はピクセル→ワールド変換から自動導出

// InteractiveCamera: 新メソッド（デフォルト実装あり）
// カスタム実装は set_viewport_size() / set_cursor_position() を
// 必要に応じてオーバーライド。
```
