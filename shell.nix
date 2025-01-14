{ pkgs ? import <nixpkgs> {} }:
  pkgs.mkShell {
    # nativeBuildInputs is usually what you want -- tools you need to run
    # One on each row to aid with version control
    nativeBuildInputs = with pkgs.buildPackages; [
      ffmpeg
      sox
      uv
    ];
}
