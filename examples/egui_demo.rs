use mikage::{App, FrameContext, RunConfig, UpdateContext};

struct EguiDemoApp {
    counter: i32,
    name: String,
    slider_value: f32,
}

impl App for EguiDemoApp {
    fn update(&mut self, _ctx: &mut UpdateContext) {}

    fn encode(&mut self, ctx: &mut FrameContext) {
        let _pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.surface_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.1,
                        b: 0.15,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
    }

    fn gui(&mut self, egui_ctx: &mikage::egui::Context) {
        mikage::egui::Window::new("mikage egui demo").show(egui_ctx, |ui| {
            ui.heading("Hello from mikage!");
            ui.horizontal(|ui| {
                ui.label("Your name:");
                ui.text_edit_singleline(&mut self.name);
            });
            ui.add(mikage::egui::Slider::new(&mut self.slider_value, 0.0..=100.0).text("value"));
            ui.horizontal(|ui| {
                if ui.button("-").clicked() {
                    self.counter -= 1;
                }
                ui.label(format!("Counter: {}", self.counter));
                if ui.button("+").clicked() {
                    self.counter += 1;
                }
            });
            ui.label(format!("Hello, {}!", self.name));
        });
    }
}

fn main() {
    mikage::run(
        |_ctx, _size| EguiDemoApp {
            counter: 0,
            name: "World".to_string(),
            slider_value: 42.0,
        },
        RunConfig {
            title: "mikage - egui demo".to_string(),
            ..Default::default()
        },
    );
}
