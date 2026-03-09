{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
    packages = [
        (pkgs.python3.withPackages (ps: with ps; [
            numpy
            pandas
            scikit-learn
            statsmodels
            matplotlib
            pip
            virtualenv
        ]))
        pkgs.csvkit
    ];

    shellHook = ''
      python -m venv .venv
      source .venv/bin/activate
      pip install optuna optuna-integration pmdarima
    '';
}
