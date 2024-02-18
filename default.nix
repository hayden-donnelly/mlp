{
    stdenv,
    libgcc,
    ...
}:
stdenv.mkDerivation {
    pname = "cpu-mlp";
    version = "v0.0.0";
    src = ./src;
    buildInputs = [ 
        libgcc
    ];
    buildPhase = ''
        gcc main.c -o cpu-mlp
    '';
    installPhase = ''
        mkdir -p $out/bin
        mv cpu-mlp $out/bin/cpu-mlp
    '';
}
