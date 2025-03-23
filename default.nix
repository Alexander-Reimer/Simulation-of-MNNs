{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = [ ];
  nativeBuildInputs = with pkgs; [ julia ];
  # load correct (current) project environment automatically
  JULIA_PROJECT = "@.";
  # adding julia lib path to "global" LD_LIBRARY_PATH causes issues as those
  # libs are chosen over correct ones, so it is instead only added when running
  # the julia command
  shellHook = ''
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib:/run/opengl-driver-32/lib"
    alias julia='LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.julia}/lib/julia" julia --threads=auto'
  '';
}
