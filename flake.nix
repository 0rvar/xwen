{
  description = "Devshell for the maxuna engine (M5 Max / Metal only)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { self, nixpkgs }:
    let
      # Single-system on purpose: maxuna targets this machine only.
      pkgs = import nixpkgs { system = "aarch64-darwin"; };
    in
    {
      # mkShellNoCC, not mkShell: the darwin stdenv exports SDKROOT pointing at
      # nixpkgs' Apple SDK, and a cargo build under that SDKROOT links maxuna
      # against it. The Metal runtime compiler derives its default MSL version
      # from the SDK the binary was LINKED against, so an old nix SDK silently
      # breaks the Metal-4 tensor kernels (f16_t/mm_id) at runtime. Builds must
      # use the system CLT SDK.
      devShells.aarch64-darwin.default = pkgs.mkShellNoCC {
        shellHook = ''
          lefthook install
        '';
        # Rust (cargo/rustfmt) comes from the nix-darwin system profile, not
        # this shell — a second rustfmt version here could disagree with it.
        buildInputs = with pkgs; [
          lefthook
        ];
      };
    };
}
