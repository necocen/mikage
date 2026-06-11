//! Lightweight WGSL shader preprocessor with `#import` resolution.
//!
//! Resolves `#import module::name` directives by hoisting registered module
//! sources to the top of the output. Each module is expanded exactly once,
//! dependencies are resolved recursively, and circular imports are detected.

use std::collections::{HashMap, HashSet};
use std::fmt;

/// WGSL source for the shared `SceneUniform` struct.
pub const SCENE_TYPES_WGSL: &str = include_str!("../assets/shaders/scene_types.wgsl");

/// WGSL source for math constants and utilities (PI, TAU, rotate2d, clamp_magnitude).
pub const MATH_WGSL: &str = include_str!("../assets/shaders/math.wgsl");

/// WGSL source for color space conversion utilities (HSV, HSL, sRGB).
pub const COLOR_UTILS_WGSL: &str = include_str!("../assets/shaders/color_utils.wgsl");

/// WGSL source for basic lighting utilities (Lambert diffuse).
pub const LIGHTING_WGSL: &str = include_str!("../assets/shaders/lighting.wgsl");

/// Errors that can occur during shader import resolution.
#[derive(Debug, Clone)]
pub enum ShaderError {
    /// A referenced module was not registered.
    ModuleNotFound { name: String },
    /// A circular dependency was detected.
    CircularDependency { chain: Vec<String> },
}

impl fmt::Display for ShaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShaderError::ModuleNotFound { name } => write!(f, "module not found: {name}"),
            ShaderError::CircularDependency { chain } => {
                write!(f, "circular dependency: {}", chain.join(" -> "))
            }
        }
    }
}

impl std::error::Error for ShaderError {}

/// A lightweight WGSL preprocessor that resolves `#import` directives.
///
/// Register named modules, then call [`resolve`](ShaderProcessor::resolve) to
/// expand all imports. Expanded modules are hoisted to the top of the output
/// in dependency order.
pub struct ShaderProcessor {
    modules: HashMap<String, String>,
}

impl Default for ShaderProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl ShaderProcessor {
    /// Creates a new `ShaderProcessor` with built-in `mikage::*` modules pre-registered.
    pub fn new() -> Self {
        let mut sp = Self {
            modules: HashMap::new(),
        };
        sp.register("mikage::scene_types", SCENE_TYPES_WGSL);
        sp.register("mikage::math", MATH_WGSL);
        sp.register("mikage::color_utils", COLOR_UTILS_WGSL);
        sp.register("mikage::lighting", LIGHTING_WGSL);
        sp
    }

    /// Registers a named module. Chainable.
    ///
    /// The name should be hierarchical, e.g. `"mikage::scene_types"`.
    pub fn register(&mut self, name: &str, source: &str) -> &mut Self {
        self.modules.insert(name.to_string(), source.to_string());
        self
    }

    /// Resolves all `#import` directives in `source`.
    ///
    /// Imported modules are hoisted to the top (in dependency order), wrapped
    /// in `// --- begin/end ---` comments. Each module is expanded at most once.
    pub fn resolve(&self, source: &str) -> Result<String, ShaderError> {
        let mut expanded = Vec::new();
        let mut visited = HashSet::new();
        let mut stack = Vec::new();

        // Collect imports from the root source
        let imports = Self::collect_imports(source);

        // Recursively resolve each import
        for import_name in &imports {
            self.resolve_module(import_name, &mut expanded, &mut visited, &mut stack)?;
        }

        // Build the hoisted block
        let mut result = String::new();
        for (name, body) in &expanded {
            result.push_str(&format!("// --- begin {name} ---\n"));
            result.push_str(body);
            if !body.ends_with('\n') {
                result.push('\n');
            }
            result.push_str("// --- end ---\n");
        }

        // Strip #import and #define_import_path lines from the root source
        for (line, active_line) in Self::with_comment_stripped_lines(source) {
            let trimmed = active_line.trim_start();
            if trimmed.starts_with("#import") || trimmed.starts_with("#define_import_path") {
                continue;
            }
            result.push_str(line);
            result.push('\n');
        }

        Ok(result)
    }

    /// Collects `#import` module names from a source string.
    fn collect_imports(source: &str) -> Vec<String> {
        let mut imports = Vec::new();
        for (_, active_line) in Self::with_comment_stripped_lines(source) {
            let line = active_line.as_str();
            let trimmed = line.trim_start();
            if let Some(rest) = trimmed.strip_prefix("#import") {
                let rest = rest.trim();
                // Strip selective imports: `module::{Item1, Item2}` -> `module`
                let token = rest.split_whitespace().next().unwrap_or("");
                let name = if let Some(idx) = token.find("::{") {
                    &token[..idx]
                } else {
                    token
                };
                if !name.is_empty() {
                    imports.push(name.to_string());
                }
            }
        }
        imports
    }

    fn with_comment_stripped_lines(source: &str) -> Vec<(&str, String)> {
        let stripped = Self::strip_comments_preserve_newlines(source);
        let stripped_lines: Vec<&str> = stripped.split('\n').collect();
        source
            .lines()
            .enumerate()
            .map(|(index, line)| {
                (
                    line,
                    stripped_lines
                        .get(index)
                        .copied()
                        .unwrap_or_default()
                        .to_string(),
                )
            })
            .collect()
    }

    fn strip_comments_preserve_newlines(source: &str) -> String {
        let mut out = String::with_capacity(source.len());
        let mut chars = source.chars().peekable();
        let mut in_block = false;

        while let Some(ch) = chars.next() {
            if in_block {
                if ch == '\n' {
                    out.push('\n');
                } else if ch == '*' && chars.peek() == Some(&'/') {
                    chars.next();
                    in_block = false;
                }
                continue;
            }

            if ch == '/' {
                match chars.peek() {
                    Some('/') => {
                        chars.next();
                        for next in chars.by_ref() {
                            if next == '\n' {
                                out.push('\n');
                                break;
                            }
                        }
                    }
                    Some('*') => {
                        chars.next();
                        in_block = true;
                    }
                    _ => out.push(ch),
                }
            } else {
                out.push(ch);
            }
        }

        out
    }

    /// Recursively resolves a module and its dependencies.
    fn resolve_module(
        &self,
        name: &str,
        expanded: &mut Vec<(String, String)>,
        visited: &mut HashSet<String>,
        stack: &mut Vec<String>,
    ) -> Result<(), ShaderError> {
        // Already expanded — skip
        if visited.contains(name) {
            return Ok(());
        }

        // Circular dependency check
        if stack.contains(&name.to_string()) {
            let mut chain: Vec<String> = stack
                .iter()
                .skip_while(|s| s.as_str() != name)
                .cloned()
                .collect();
            chain.push(name.to_string());
            return Err(ShaderError::CircularDependency { chain });
        }

        let source = self
            .modules
            .get(name)
            .ok_or_else(|| ShaderError::ModuleNotFound {
                name: name.to_string(),
            })?;

        stack.push(name.to_string());

        // Resolve transitive imports first
        let sub_imports = Self::collect_imports(source);
        for sub in &sub_imports {
            self.resolve_module(sub, expanded, visited, stack)?;
        }

        stack.pop();

        // Strip #import / #define_import_path lines from the module body
        let clean: String = Self::with_comment_stripped_lines(source)
            .into_iter()
            .filter_map(|(line, active_line)| {
                let t = active_line.trim_start();
                (!t.starts_with("#import") && !t.starts_with("#define_import_path")).then_some(line)
            })
            .collect::<Vec<_>>()
            .join("\n");

        visited.insert(name.to_string());
        expanded.push((name.to_string(), clean));

        Ok(())
    }
}

/// Returns a [`ShaderProcessor`] with built-in mikage modules pre-registered.
pub(crate) fn mikage_shader_processor() -> ShaderProcessor {
    ShaderProcessor::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_import() {
        let mut sp = ShaderProcessor::new();
        sp.register("math", "fn add(a: f32, b: f32) -> f32 { return a + b; }");

        let source = "#import math\nfn main() { let x = add(1.0, 2.0); }";
        let result = sp.resolve(source).unwrap();

        assert!(result.contains("// --- begin math ---"));
        assert!(result.contains("fn add("));
        assert!(result.contains("// --- end ---"));
        assert!(result.contains("fn main()"));
        assert!(!result.contains("#import"));
    }

    #[test]
    fn recursive_import() {
        let mut sp = ShaderProcessor::new();
        sp.register("base", "struct Foo { x: f32 };");
        sp.register("mid", "#import base\nfn use_foo() {}");

        let source = "#import mid\nfn main() {}";
        let result = sp.resolve(source).unwrap();

        // base should come before mid
        let base_pos = result.find("// --- begin base ---").unwrap();
        let mid_pos = result.find("// --- begin mid ---").unwrap();
        assert!(base_pos < mid_pos);
    }

    #[test]
    fn circular_dependency() {
        let mut sp = ShaderProcessor::new();
        sp.register("a", "#import b\nstruct A {};");
        sp.register("b", "#import a\nstruct B {};");

        let source = "#import a";
        let result = sp.resolve(source);

        assert!(matches!(
            result,
            Err(ShaderError::CircularDependency { .. })
        ));
    }

    #[test]
    fn duplicate_import_expanded_once() {
        let mut sp = ShaderProcessor::new();
        sp.register("shared", "struct S { x: f32 };");

        let source = "#import shared\n#import shared\nfn main() {}";
        let result = sp.resolve(source).unwrap();

        let count = result.matches("// --- begin shared ---").count();
        assert_eq!(count, 1);
    }

    #[test]
    fn module_not_found() {
        let sp = ShaderProcessor::new();
        let source = "#import nonexistent";
        let result = sp.resolve(source);

        assert!(matches!(result, Err(ShaderError::ModuleNotFound { .. })));
    }

    #[test]
    fn selective_import_stripped() {
        let mut sp = ShaderProcessor::new();
        sp.register(
            "granular_clock::physics_types",
            "struct Particle { pos: vec3<f32> };\nstruct Params { dt: f32 };",
        );

        let source = "#import granular_clock::physics_types::{Particle, Params}\nfn main() {}";
        let result = sp.resolve(source).unwrap();

        assert!(result.contains("struct Particle"));
        assert!(result.contains("struct Params"));
        assert!(!result.contains("#import"));
    }

    #[test]
    fn selective_import_with_deep_module_path() {
        let mut sp = ShaderProcessor::new();
        sp.register("a::b::c", "struct Deep { x: f32 };");

        let source = "#import a::b::c::{Deep}\nfn main() {}";
        let result = sp.resolve(source).unwrap();

        assert!(result.contains("// --- begin a::b::c ---"));
        assert!(result.contains("struct Deep"));
    }

    #[test]
    fn ignores_imports_inside_block_comments() {
        let sp = ShaderProcessor::new();
        let source = "/*\n#import missing\n*/\nfn main() {}";
        let result = sp.resolve(source).unwrap();

        assert!(result.contains("#import missing"));
        assert!(result.contains("fn main()"));
    }

    #[test]
    fn define_import_path_stripped() {
        let mut sp = ShaderProcessor::new();
        sp.register("mymod", "#define_import_path mymod\nstruct Foo { x: f32 };");

        let source = "#import mymod\nfn main() {}";
        let result = sp.resolve(source).unwrap();

        assert!(result.contains("struct Foo"));
        assert!(!result.contains("#define_import_path"));
    }

    #[test]
    fn hoist_before_body() {
        let mut sp = ShaderProcessor::new();
        sp.register("types", "struct T { x: f32 };");

        let source = "// header\n#import types\nfn main() {}";
        let result = sp.resolve(source).unwrap();

        let types_pos = result.find("struct T").unwrap();
        let main_pos = result.find("fn main()").unwrap();
        assert!(types_pos < main_pos);
    }

    #[test]
    fn diamond_dependency() {
        let mut sp = ShaderProcessor::new();
        sp.register("base", "struct Base {};");
        sp.register("left", "#import base\nstruct Left {};");
        sp.register("right", "#import base\nstruct Right {};");

        let source = "#import left\n#import right\nfn main() {}";
        let result = sp.resolve(source).unwrap();

        // base expanded exactly once
        assert_eq!(result.matches("// --- begin base ---").count(), 1);
        // All three present
        assert!(result.contains("struct Base"));
        assert!(result.contains("struct Left"));
        assert!(result.contains("struct Right"));
    }
}
