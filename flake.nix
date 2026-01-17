{
  description = "Development environment for tenxor";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        cargoToml = pkgs.lib.importTOML ./Cargo.toml;
      in
      {
        formatter = pkgs.nixfmt-tree;

        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = cargoToml.package.name;
          version = cargoToml.package.version;

          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          nativeBuildInputs = with pkgs; [
            cudaPackages.cuda_nvcc
            llvmPackages_19.libcxx
          ];

          buildInputs = with pkgs; [
            bintools
            cudaPackages.cuda_cudart
            cudaPackages.nvcomp
          ];

          # Set AR, the archiver tool
          AR = "${pkgs.bintools}/bin/ar";

          # Set NVCC_CCBIN, the host compiler for nvcc
          NVCC_CCBIN = "${pkgs.clang}/bin/clang++";

          # Set NVCC_CFLAGS, flags for the nvcc command
          # sm_86 is for Ampere architecture, change if you have a different GPU
          NVCC_CFLAGS = "-cudart=static -arch=sm_86";

          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          LIBCXX_INCLUDE_PATH = "${pkgs.llvmPackages_19.libcxx.dev}/include/c++/v1";
          CUDA_INCLUDE = "${pkgs.lib.getOutput "dev" pkgs.cudaPackages.cuda_cudart}/include";
          CUDA_NVCC_INCLUDE = "${pkgs.cudaPackages.cuda_nvcc}/include";
          CUDA_LIB = "${pkgs.lib.getOutput "static" pkgs.cudaPackages.cuda_cudart}/lib";
          NVCOMP_INCLUDE = "${pkgs.cudaPackages.nvcomp.include}/include";
          NVCOMP_LIB = "${pkgs.cudaPackages.nvcomp.static}/lib";

          meta = {
            description = "NVCOMP compression benchmark";
            mainProgram = "benchmark";
          };
        };

        devShells.default = (pkgs.mkShell.override { stdenv = pkgs.llvmPackages_19.stdenv; }) {
          name = cargoToml.package.name;

          buildInputs = with pkgs; [
            bintools
            clang-tools
            bacon
            cargo
            clippy
            cudaPackages.cuda_cudart
            cudaPackages.cuda_nvcc
            cudaPackages.nvcomp
            lldb
            llvmPackages_19.libcxx
            rust-analyzer
            rustc
            rustfmt
            taplo
          ];

          # Set AR, the archiver tool
          AR = "${pkgs.bintools}/bin/ar";

          # Set NVCC_CCBIN, the host compiler for nvcc
          NVCC_CCBIN = "${pkgs.clang}/bin/clang++";

          # Set NVCC_CFLAGS, flags for the nvcc command
          # sm_86 is for Ampere architecture, change if you have a different GPU
          NVCC_CFLAGS = "-cudart=static -arch=sm_86";

          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          LIBCXX_INCLUDE_PATH = "${pkgs.llvmPackages_19.libcxx.dev}/include/c++/v1";
          CUDA_INCLUDE = "${pkgs.lib.getOutput "dev" pkgs.cudaPackages.cuda_cudart}/include";
          CUDA_NVCC_INCLUDE = "${pkgs.cudaPackages.cuda_nvcc}/include";
          CUDA_LIB = "${pkgs.lib.getOutput "static" pkgs.cudaPackages.cuda_cudart}/lib";
          NVCOMP_INCLUDE = "${pkgs.cudaPackages.nvcomp.include}/include";
          NVCOMP_LIB = "${pkgs.cudaPackages.nvcomp.static}/lib";
        };
      }
    );
}
