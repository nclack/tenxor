use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    check_required_env_vars();
    configure_linking();
    compile_cuda();
    generate_bindings();
}

fn check_required_env_vars() {
    let required = [
        "AR",
        "CUDA_INCLUDE",
        "CUDA_LIB",
        "CUDA_NVCC_INCLUDE",
        "NVCC_CCBIN",
        "NVCC_CFLAGS",
        "NVCOMP_INCLUDE",
        "NVCOMP_LIB",
    ];

    let missing: Vec<&str> = required
        .iter()
        .filter(|var| env::var(var).is_err())
        .copied()
        .collect();

    if !missing.is_empty() {
        eprintln!("Error: Missing required environment variables:");
        for var in &missing {
            eprintln!("  - {}", var);
        }
        eprintln!("\nPlease set these environment variables in your flake.nix before building.");
        panic!("Missing required environment variables");
    }
}

fn get_env(var: &str) -> String {
    env::var(var).unwrap_or_else(|_| panic!("Required environment variable {} is not set.", var))
}

fn configure_linking() {
    println!("cargo:rustc-link-search=native={}", get_env("CUDA_LIB"));
    println!("cargo:rustc-link-search=native={}", get_env("NVCOMP_LIB"));
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=static=nvcomp_static");
    println!("cargo:rustc-link-lib=static=nvcomp_device_static");

    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=rt");
        println!("cargo:rustc-link-lib=stdc++");
    }
}

fn compile_cuda() {
    let out_dir = PathBuf::from(get_env("OUT_DIR"));
    let obj_path = out_dir.join("transpose.o");
    let lib_path = out_dir.join("libtranspose.a");

    let nvcc_cflags = get_env("NVCC_CFLAGS");

    let nvcc_output = Command::new("nvcc")
        .arg("-c")
        .arg("src/transpose.cu")
        .arg("-o")
        .arg(&obj_path)
        .arg("-I")
        .arg(get_env("CUDA_INCLUDE"))
        .arg("-ccbin")
        .arg(get_env("NVCC_CCBIN"))
        .args(nvcc_cflags.split_whitespace())
        .output()
        .expect("Failed to execute nvcc");

    if !nvcc_output.status.success() {
        panic!(
            "nvcc failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&nvcc_output.stdout),
            String::from_utf8_lossy(&nvcc_output.stderr)
        );
    }

    let ar_output = Command::new(get_env("AR"))
        .arg("rcs")
        .arg(&lib_path)
        .arg(&obj_path)
        .output()
        .expect("Failed to execute ar");

    if !ar_output.status.success() {
        panic!(
            "ar failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&ar_output.stdout),
            String::from_utf8_lossy(&ar_output.stderr)
        );
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=transpose");
    println!("cargo:rerun-if-changed=src/transpose.cu");
    println!("cargo:rerun-if-changed=src/transpose.h");
}

fn generate_bindings() {
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", get_env("CUDA_INCLUDE")))
        .clang_arg(format!("-I{}", get_env("CUDA_NVCC_INCLUDE")))
        .clang_arg(format!("-I{}", get_env("NVCOMP_INCLUDE")))
        .clang_arg(format!("-I{}", get_env("LIBCXX_INCLUDE_PATH")))
        .allowlist_function("tiled")
        .allowlist_function("nvcomp.*")
        .allowlist_type("nvcomp.*")
        .allowlist_type("cudaStream_t")
        .allowlist_type("Layout")
        .allowlist_type("Dimension")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .derive_debug(false)
        .layout_tests(false)
        .generate_comments(false)
        .generate()
        .expect("Failed to generate bindings");

    let out_path = PathBuf::from(get_env("OUT_DIR"));
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Failed to write bindings");
}

