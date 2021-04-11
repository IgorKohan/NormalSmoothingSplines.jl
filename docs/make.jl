using Documenter, NormalSmoothingSplines

makedocs(
    sitename = "NormalSmoothingSplines.jl",
	format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
	authors = "Igor Kohanovsky",
    pages = [
				"Home" => "index.md",
				"Public API" => "Public-API.md",
			]
)

deploydocs(
    repo = "github.com/IgorKohan/NormalSmoothingSplines.jl.git",
)
