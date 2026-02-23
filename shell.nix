{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
    packages = [
        (pkgs.python3.withPackages (ps: with ps; [
            numpy
            pandas
            scikit-learn
            optuna
            statsmodels
            matplotlib
        ]))
        pkgs.csvkit
    ];
}
