use std::env;
use std::path::PathBuf;

fn main() {
    if cfg!(target_os = "windows") {
        configure_windows();
    } else if cfg!(target_os = "linux") {
        configure_linux();
    } else {
        panic!("Unsupported platform");
    }

    generate_bindings();
}

fn configure_windows() {
    // Link with CUDA 12.9 and nvCOMP v4.2 libraries
    println!("cargo:rustc-link-search=native=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\lib\\x64");
    println!("cargo:rustc-link-search=native=C:\\Program Files\\NVIDIA nvCOMP\\v4.2\\lib\\12");

    // Link with static libraries
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=static=nvcomp_static");
    println!("cargo:rustc-link-lib=static=nvcomp_device_static");

    // Required system libraries for static linking
    println!("cargo:rustc-link-lib=kernel32");
    println!("cargo:rustc-link-lib=user32");
    println!("cargo:rustc-link-lib=advapi32");
}


fn find_nvcc_include_path() -> Option<String> {
    env::var("CUDA_NVCC_PATH")
        .ok()
        .map(|nvcc_path| format!("{}/include", nvcc_path))
}

fn find_cuda_include_paths() -> Vec<String> {
    let mut paths = Vec::new();

    // Get cuda runtime headers from pkg-config
    if let Ok(cuda_info) = pkg_config::Config::new().probe("cudart") {
        paths.extend(
            cuda_info
                .include_paths
                .iter()
                .map(|p| p.display().to_string()),
        );
    }

    // Also need cuda compiler headers (for crt/host_config.h, etc.)
    if let Some(nvcc_include) = find_nvcc_include_path() {
        paths.push(nvcc_include);
    }

    // Fallback to CUDA_ROOT env var or standard path
    if paths.is_empty() {
        if let Ok(cuda_root) = env::var("CUDA_ROOT") {
            paths.push(format!("{}/include", cuda_root));
        } else {
            paths.push("/usr/local/cuda/include".to_string());
        }
    }

    paths
}


fn find_nvcomp_include_paths() -> Vec<String> {
    if let Ok(result) = pkg_config::Config::new().probe("nvcomp") {
        return result
            .include_paths
            .iter()
            .map(|p| p.display().to_string())
            .collect();
    }

    if let Ok(nvcomp_include) = env::var("NVCOMP_INCLUDE") {
        return vec![nvcomp_include];
    }

    if let Ok(nvcomp_lib) = env::var("NVCOMP_LIB") {
        return vec![format!("{}/include", nvcomp_lib)];
    }

    vec![
        "/usr/local/include".to_string(),
        "/usr/include/nvcomp_12".to_string(),
    ]
}

fn configure_linux() {
    // Use pkg-config for CUDA driver library (dynamic)
    if pkg_config::Config::new().probe("cuda").is_err() {
        eprintln!("Warning: pkg-config couldn't find cuda");
    }

    // Use pkg-config for CUDA runtime library (static)
    if pkg_config::Config::new()
        .statik(true)
        .probe("cudart")
        .is_err()
    {
        eprintln!("Warning: pkg-config couldn't find cudart");
    }

    // Use pkg-config for nvCOMP library (static)
    if pkg_config::Config::new()
        .statik(true)
        .probe("nvcomp")
        .is_err()
    {
        eprintln!("Warning: pkg-config couldn't find nvcomp");
    }

    // System libraries needed for static linking
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=rt");
    println!("cargo:rustc-link-lib=stdc++");
}

fn generate_bindings() {
    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .allowlist_type("nvcompStatus_t")
        .allowlist_type("cudaStream_t")
        .allowlist_type("nvcompBatchedZstdCompressOpts_t")
        .allowlist_type("nvcompBatchedZstdDecompressOpts_t")
        .allowlist_type("nvcompBatchedLZ4CompressOpts_t")
        .allowlist_type("nvcompBatchedLZ4DecompressOpts_t")
        .allowlist_type("nvcompType_t")
        .allowlist_type("nvcompDecompressBackend_t")
        .allowlist_function("nvcompBatchedZstdDecompressAsync")
        .allowlist_function("nvcompBatchedZstdDecompressGetTempSizeAsync")
        .allowlist_function("nvcompBatchedZstdCompressAsync")
        .allowlist_function("nvcompBatchedZstdCompressGetTempSizeAsync")
        .allowlist_function("nvcompBatchedLZ4DecompressAsync")
        .allowlist_function("nvcompBatchedLZ4DecompressGetTempSizeAsync")
        .allowlist_function("nvcompBatchedLZ4CompressAsync")
        .allowlist_function("nvcompBatchedLZ4CompressGetTempSizeAsync")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Silence naming convention warnings
        .derive_debug(false)
        .layout_tests(false)
        .generate_comments(false);

    // Add platform-specific include paths
    if cfg!(target_os = "windows") {
        builder = builder
            .clang_arg("-IC:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\include")
            .clang_arg("-IC:\\Program Files\\NVIDIA nvCOMP\\v4.2\\include");
    } else if cfg!(target_os = "linux") {
        for path in find_cuda_include_paths() {
            builder = builder.clang_arg(format!("-I{}", path));
        }

        for path in find_nvcomp_include_paths() {
            builder = builder.clang_arg(format!("-I{}", path));
        }
    }

    let bindings = builder
        .generate()
        .expect("Failed to generate nvCOMP bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Failed to write bindings");
}
