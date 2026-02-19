use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=frontend/src");
    println!("cargo:rerun-if-changed=frontend/package.json");

    if std::env::var("CARGO_FEATURE_EMBEDDED_FRONTEND").is_ok() {
        build_frontend();
    }
}

fn build_frontend() {
    let frontend_dir = std::path::Path::new("frontend");
    if !frontend_dir.exists() {
        panic!("frontend directory not found");
    }

    // Find bun or npm
    let (pkg_mgr, install_args, build_args) = if which::which("bun").is_ok() {
        ("bun", vec!["install"], vec!["run", "build"])
    } else if which::which("npm").is_ok() {
        ("npm", vec!["install"], vec!["run", "build"])
    } else {
        panic!("bun or npm required to build frontend");
    };

    // Install dependencies
    let status = Command::new(pkg_mgr)
        .args(&install_args)
        .current_dir(frontend_dir)
        .status()
        .expect("failed to install frontend dependencies");

    if !status.success() {
        panic!("frontend dependency installation failed");
    }

    // Build frontend
    let status = Command::new(pkg_mgr)
        .args(&build_args)
        .current_dir(frontend_dir)
        .status()
        .expect("failed to build frontend");

    if !status.success() {
        panic!("frontend build failed");
    }
}
