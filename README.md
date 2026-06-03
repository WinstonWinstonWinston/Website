# Winston Sullivan — Personal Website

Academic portfolio site for Harry Winston Sullivan, PhD student in Chemical Engineering at the University of Minnesota.

**Live:** [winstonsullivan.netlify.app](https://winstonsullivan.netlify.app)

## Stack

- [Hugo](https://gohugo.io/) static site generator with [Wowchemy](https://wowchemy.com/) v5 academic theme
- Custom dark purple theme with Brownian motion canvas animation
- Deployed via [Netlify](https://www.netlify.com/) on push to `main`

## Local Development

```
hugo server
```

Requires Hugo v0.108.0+.

## Structure

```
config/_default/    Site configuration (theme, nav, params)
content/home/       Homepage widget sections
content/authors/    Author profile
content/project/    Project portfolio entries
content/publication/ Academic publications
static/uploads/     PDFs (CV, papers, presentations)
static/js/          Brownian motion animation
assets/scss/        Custom styling
data/themes/        Custom color theme
data/fonts/         Custom font config
```

## License

Content is the author's own. Site framework is MIT licensed (Wowchemy).
