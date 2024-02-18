mkdir -p build &&
rm -f build/cpu_mlp &&
gcc -o cpu_mlp src/main.c -lm &&
mv cpu_mlp build/cpu_mlp &&
./build/cpu_mlp
