{
    description = "Multilayer perceptron training on a CPU.";

    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
        flake-utils.url = "github:numtide/flake-utils";
    };

    outputs = inputs@{ self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system: let
        inherit (nixpkgs);
        pkgs = import nixpkgs { inherit system; };
    in {
        devShells = {
            default = pkgs.mkShell {
                buildInputs = with pkgs; [ libgcc ];
            };
        };
    });
}
