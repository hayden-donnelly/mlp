{
    description = "MLP training";

    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
        nixgl.url = "github:nix-community/nixGL";
        flake-utils.url = "github:numtide/flake-utils";
    };
    outputs = inputs@{ 
        self, 
        nixpkgs, 
        nixgl,
        flake-utils, 
        ... 
    }: flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system: 
        let
            inherit (nixpkgs) lib;
            pkgs = import nixpkgs {
                inherit system;
                overlays = [
                    nixgl.overlay 
                ];
                config = {
                    allowUnfree = true;
                    cudaSupport = true;
                };
            };
        in {
            devShells.default = (with pkgs; mkShell.override { stdenv = gcc12Stdenv; }) {
                name = "cuda";
                buildInputs = with pkgs; [
                    stdenv.cc.cc
                    clang-tools
                    cudaPackages.cudatoolkit
                    cudaPackages.cuda_cudart
                    cudaPackages.cudnn
                    gcc12
                ];
                shellHook = ''
                    export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
                    source <(sed -Ee '/\$@/d' ${lib.getExe pkgs.nixgl.nixGLIntel})
                    source <(sed -Ee '/\$@/d' ${lib.getExe pkgs.nixgl.auto.nixGLNvidia}*)
                '';
            };
        }
    );
}
