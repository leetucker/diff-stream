# Data sources

Most test panels here are simulated by `diff_stream.simulate` or by the
reference scripts in `benchmarks/external/`. The files below come from
published applications. Each is kept as it was distributed, apart from sorting
and, where noted, columns added for the tests.

## mpdta.csv

County-level teen employment, 500 counties, 2003–2007: a subset of the data
used in Callaway, B. and P. H. C. Sant'Anna (2021), "Difference-in-Differences
with Multiple Time Periods," *Journal of Econometrics* 225(2), 200–230, as
distributed with R's `did` package (`data(mpdta)`). The authors built it from
the U.S. Census Bureau's Quarterly Workforce Indicators; the treatment date
(`first.treat`) is the year the county's state raised its minimum wage, and
`lpop` is the log of county population in thousands.

Exported by `benchmarks/external/mpdta_reference.R`, with the original column
names and calendar years.

## castle.csv and castle_source/

State homicide rates and castle doctrine laws, 50 states, 2000–2010, from
Cheng, C. and M. Hoekstra (2013), "Does Strengthening Self-Defense Law Deter
Crime or Escalate Violence? Evidence from Expansions to Castle Doctrine,"
*Journal of Human Resources* 48(3), 821–854, as distributed with R's `did2s`
package. `castle_source/` holds the distributed `castle.rda` with did2s's
DESCRIPTION, LICENSE and the commit it was downloaded from.

## bacon_castle_covariates.csv

The same Cheng and Hoekstra (2013) panel as distributed with R's `bacondecomp`
package (`data(castle)`), with the columns the Goodman-Bacon covariate tests
use. `benchmarks/external/bacon_extended_reference.R` adds `id`, `time`,
`first_treat`, `y`, `x`, `z`, `unit_w` and `w` as copies or simple functions of
the distributed columns.

## math_reform.csv

State-by-cohort data on high school math graduation requirements, an
aggregated version of the data in Goodman, J. (2019), "The Labor of Division:
Returns to Compulsory High School Math Coursework," *Journal of Labor
Economics* 37(4), 1141–1182, as distributed with R's `bacondecomp` package
(`data(math_reform)`). Exported by
`benchmarks/external/math_reform_reference.R`.

## twfe_wagepan.csv

The panel of young men's wages and union membership, 1980–1987, from Vella,
F. and M. Verbeek (1998), "Whose Wages Do Unions Raise? A Dynamic Model of
Unionism and Wage Rate Determination for Young Men," *Journal of Applied
Econometrics* 13(2), 163–183, widely distributed as `wagepan` with
Wooldridge's econometrics textbooks. This copy, with first-difference columns
(`diff_lwage`, `diff_union`) added, is the example file distributed by de
Chaisemartin and D'Haultfœuille in their
[twowayfeweights repository](https://github.com/Credible-Answers/twowayfeweights).
