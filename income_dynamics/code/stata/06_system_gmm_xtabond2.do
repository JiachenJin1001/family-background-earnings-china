*----------------------------------------------------------*
* 06_system_gmm_xtabond2.do
* Arellano-Bond estimates of earnings persistence with xtabond2:
*   (a) Difference GMM, two-step, Windmeijer-corrected standard errors,
*       collapsed instruments lagged two and three waves, with the
*       interaction of lagged earnings and HighParentOcc14.
*   (b) System GMM (Blundell-Bond), same instruments.
* Sample: child-year rows of panel_resid with non-missing y_tilde_w and
*   HighParentOcc14 (children aged 22 to 55, waves 2014 to 2022, at
*   least three waves with positive income).
* Reads : data/intermediate/panel_resid.dta
* Writes: 06_system_gmm_xtabond2.log (this folder)
* Requires: xtabond2 (ssc install xtabond2)
*----------------------------------------------------------*
version 17
clear all
set more off

* Run from this folder (income_dynamics/code/stata); the pipeline folder is two levels up.
global WSM "../.."
capture log close
log using "06_system_gmm_xtabond2.log", replace text

use "$WSM/data/intermediate/panel_resid.dta", clear
drop if missing(y_tilde_w) | missing(HighParentOcc14)
destring pid year HighParentOcc14, replace force
xtset pid year, delta(2)

gen double y_hp = y_tilde_w * HighParentOcc14

di as result _n "=== (a) Difference GMM: two-step, Windmeijer, collapsed lags 2-3 ==="
xtabond2 y_tilde_w L.y_tilde_w L.y_hp,              ///
    gmm(y_tilde_w, lag(2 3) collapse eq(diff))      ///
    gmm(y_hp,      lag(2 3) collapse eq(diff))      ///
    noleveleq twostep robust noconstant
di as result "rho_0 = " %7.4f _b[L.y_tilde_w] "  (SE " %6.4f _se[L.y_tilde_w] ")"
di as result "rho_1 = " %7.4f _b[L.y_hp]      "  (SE " %6.4f _se[L.y_hp] ")"
di as result "Hansen J = " %6.3f e(hansen) "  df = " e(hansendf) ///
    "  p = " %5.3f e(hansenp)
di as result "instruments = " e(j)

di as result _n "=== (b) System GMM: two-step, Windmeijer, collapsed lags 2-3 ==="
xtabond2 y_tilde_w L.y_tilde_w L.y_hp,              ///
    gmm(y_tilde_w, lag(2 3) collapse)               ///
    gmm(y_hp,      lag(2 3) collapse)               ///
    twostep robust h(2)
di as result "rho_0 = " %7.4f _b[L.y_tilde_w] "  (SE " %6.4f _se[L.y_tilde_w] ")"
di as result "rho_1 = " %7.4f _b[L.y_hp]      "  (SE " %6.4f _se[L.y_hp] ")"
di as result "Hansen J = " %6.3f e(hansen) "  df = " e(hansendf) ///
    "  p = " %5.3f e(hansenp)
di as result "instruments = " e(j)
di as result "Difference-in-Hansen test of the level moments: see the xtabond2 output above."

log close
di as result _n "Done. Log: $WSM/code/stata/06_system_gmm_xtabond2.log"
