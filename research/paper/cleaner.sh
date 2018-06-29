infile='draft-descope.tex'
outfile='draft.tex'

sed -e "s/{\\textbackslash}'\\\\{i\\\\}/\\'{i}/g" references.bib | grep -E -v '^\W*(urldate|abstract|file|keywords)' > references.bib.tmp
mv references.bib.tmp references.bib

grep -E -v '^(%|[[:blank:]]*%|\\COMMENT)' "$infile" | fold -w80 -s > "$outfile"
