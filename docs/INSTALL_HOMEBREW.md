# Installation type Homebrew (macOS)

MLXServe n’est pas encore publié sur un tap Homebrew officiel. Ce fichier donne un **modèle de formule** pour packager votre build local ou CI.

## Prérequis

- Python **3.12+** installé (ou géré par la formule).
- Dépendance optionnelle **mlx** : `pip install -e ".[mlx]"` reste souvent le plus simple sur Apple Silicon.

## Exemple de formule (à adapter)

```ruby
class Mlxserve < Formula
  desc "OpenAI-compatible MLX inference server for Apple Silicon"
  homepage "https://github.com/ROSITO/SoloMLX-server"
  url "https://github.com/ROSITO/SoloMLX-server/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "REMPLACER_PAR_LE_SHA_DU_TARBALL"
  license "MIT"

  depends_on "python@3.12"

  def install
    venv = virtualenv_create(libexec, "python3.12")
    venv.pip_install_and_link buildpath
    bin.install_symlink libexec/"bin/mlxserve"
  end

  test do
    system bin/"mlxserve", "--help"
  end
end
```

En pratique, beaucoup d’équipes préfèrent **`pip install`** ou un script `scripts/start_server.sh` jusqu’à ce qu’une release semver et un tap dédiés soient stabilisés.
