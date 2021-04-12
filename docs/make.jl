using Documenter, NormalSmoothingSplines

makedocs(
    sitename = "NormalSmoothingSplines.jl",
	format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
	authors = "Igor Kohanovsky",
    pages = [
				"Home" => "index.md",
				"Public API" => "Public-API.md",
				"Example Usage" => "Usage.md",
				"Normal Splines Method" => "Normal-Splines-Method.md",
			]
)

deploydocs(
repo = "github.com/IgorKohan/NormalSmoothingSplines.jl.git",
devurl = "v0.1.0",
versions = ["stable" => "v^", "v#.#.#"],
)
