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
            pkg-config
            cudaPackages.cuda_nvcc
          ];

          buildInputs = with pkgs; [
            cudaPackages.cuda_cudart
            cudaPackages.nvcomp
          ];

          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          NVCOMP_LIB = "${pkgs.cudaPackages.nvcomp.static}";
          NVCOMP_INCLUDE = "${pkgs.cudaPackages.nvcomp.include}/include";
          CUDA_NVCC_PATH = "${pkgs.cudaPackages.cuda_nvcc}";

          meta = {
            description = "NVCOMP compression benchmark";
            mainProgram = "benchmark";
          };
        };

        devShells.default = (pkgs.mkShell.override { stdenv = pkgs.clangStdenv; }) {
          name = cargoToml.package.name;

          buildInputs = with pkgs; [
            clang-tools
            bacon
            cargo
            clippy
            cudaPackages.cuda_cudart
            cudaPackages.cuda_nvcc
            cudaPackages.nvcomp
            lldb
            llvmPackages.libcxx
            pkg-config
            rust-analyzer
            rustc
            rustfmt
            taplo
          ];

          CPATH = "${pkgs.llvmPackages.libcxx.dev}/include/c++/v1";
          CPLUS_INCLUDE_PATH = "${pkgs.llvmPackages.libcxx.dev}/include/c++/v1";

          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          NVCOMP_LIB = "${pkgs.cudaPackages.nvcomp.static}";
          NVCOMP_INCLUDE = "${pkgs.cudaPackages.nvcomp.include}/include";
          CUDA_NVCC_PATH = "${pkgs.cudaPackages.cuda_nvcc}";
        };
      }
    );
}
