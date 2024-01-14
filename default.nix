{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [ ];

  shellHook = ''
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib:/run/opengl-driver-32/lib"
  '';
}