*----------------------------------------------------------*
* 05_method2_abgmm.do
*   Dynamic panel model of residual log income, estimated by
*   Arellano-Bond GMM. Stata version of 05_method2_abgmm.py.
*
*   Sample: child-year rows of panel_resid with non-missing y_tilde_w
*   and HighParentOcc14 (children aged 22 to 55, waves 2014 to 2022,
*   at least three waves with positive income).
*
*   Specification:
*     - Equation in first differences for the winsorized residual
*       y_tilde_w.
*     - Regressors (in first differences): L.y_tilde_w and
*       L.y_tilde_w x HighParentOcc14.
*     - No year indicators in the equation: year effects are removed
*       from y_tilde_w when the residual is constructed.
*     - Instruments: levels lagged two and three waves, collapsed
*       (Roodman 2009), and their interactions with HighParentOcc14.
*     - One-step weight matrix W1 = (Z'Z)^{-1}.
*     - Two-step weight matrix W2 = inverse of the sum over children
*       of the outer products of the scores at the one-step residuals.
*     - Standard errors clustered by child (pid), sandwich formula.
*     - Hansen J with the two-step weight matrix and residuals.
*
*   The estimator is coded in Mata with the same formulas as the Python
*   script. xtabond2 is run for comparison. It uses a different one-step
*   weight matrix (the H matrix of Arellano and Bond) and here includes
*   year indicators, so its point estimates differ from the Mata
*   estimates.
*
*   Reads:  data/intermediate/panel_resid.dta
*           data/intermediate/method1_results.csv  (minimum-distance
*           estimates, for the value of rho_1 implied by the variance shares)
*   Writes: data/intermediate/method2_results_stata.dta
*           data/intermediate/method2_results.dta
*----------------------------------------------------------*
version 17
clear all
set more off

* BASE_INT is the folder data/intermediate of this package, relative to
* the folder of this do-file.
global BASE_INT "../../data/intermediate"

use "$BASE_INT/panel_resid.dta", clear
drop if missing(y_tilde_w) | missing(HighParentOcc14)
destring pid year HighParentOcc14, replace force
xtset pid year, delta(2)

* Number of children with at least three valid waves
bys pid: gen byte _has_obs = !missing(y_tilde_w)
bys pid: egen int n_valid = total(_has_obs)
qui count if n_valid >= 3
local n_eff = r(N)
qui distinct pid if n_valid >= 3
local n_eff_pid = r(ndistinct)
di as result _n "Effective N for AB-GMM (children with >=3 valid waves): " ///
    "`n_eff_pid' children, `n_eff' obs"

*----------------------------------------------------------*
* 1. Within-groups estimate of the AR(1) coefficient
*----------------------------------------------------------*
xtreg y_tilde_w l.y_tilde_w, fe cluster(pid)
matrix b_wg = e(b)
local rho_wg = b_wg[1, "L.y_tilde_w"]
local se_wg  = sqrt(e(V)["L.y_tilde_w","L.y_tilde_w"])
local n_wg   = e(N)
di as result _n "[Within-Groups]   rho_WG = " %6.4f `rho_wg' ///
    "  (SE " %6.4f `se_wg' ")"
di as result "  Nickell approximation of the bias for true rho=0.5, T=5: -(1+0.5)/4 = -0.375"

*----------------------------------------------------------*
* 2. First-differenced equations and collapsed instruments lagged two
*    and three waves (as in build_AB_data of 05_method2_abgmm.py)
*----------------------------------------------------------*
egen wave_idx = group(year)
qui summ wave_idx
local T = r(max)
di as result _n "Building AB-GMM equations: T = `T' waves (wave_idx 1..`T')"

* HighParentOcc14 by child (constant within child)
bys pid (wave_idx): gen byte HP14_pid = HighParentOcc14[1]

* Reshape to wide format and save to a temporary file
tempfile widefile
keep pid wave_idx y_tilde_w HP14_pid
reshape wide y_tilde_w, i(pid HP14_pid) j(wave_idx)
save `widefile', replace

* Equations for t = 3 to T, appended one wave at a time to eq_pool.
tempfile eq_pool
local first = 1
* The instrument at lag L of the equation for wave t is the level at wave
* t - L: lag 2 is y_tilde_w(t-2), lag 3 is y_tilde_w(t-3).
forvalues t = 3/`T' {
    local t1   = `t' - 1
    local t2   = `t' - 2
    local zL2  = `t' - 2   /* z_lag2 wave index */
    local zL3  = `t' - 3   /* z_lag3 wave index */
    use `widefile', clear
    keep if !missing(y_tilde_w`t', y_tilde_w`t1', y_tilde_w`t2')
    gen double Dy           = y_tilde_w`t'  - y_tilde_w`t1'
    gen double Dy_lag       = y_tilde_w`t1' - y_tilde_w`t2'
    gen double Dy_lag_x_hp  = Dy_lag * HP14_pid
    if `zL2' >= 1 {
        gen double z_lag2 = cond(missing(y_tilde_w`zL2'), 0, y_tilde_w`zL2')
    }
    else {
        gen double z_lag2 = 0
    }
    if `zL3' >= 1 {
        gen double z_lag3 = cond(missing(y_tilde_w`zL3'), 0, y_tilde_w`zL3')
    }
    else {
        gen double z_lag3 = 0
    }
    gen double z_lag2_hp = z_lag2 * HP14_pid
    gen double z_lag3_hp = z_lag3 * HP14_pid
    gen int t_idx = `t'
    keep pid HP14_pid t_idx Dy Dy_lag Dy_lag_x_hp z_lag2 z_lag3 z_lag2_hp z_lag3_hp
    if `first' == 1 {
        save `eq_pool', replace
        local first = 0
    }
    else {
        append using `eq_pool'
        save `eq_pool', replace
    }
}

use `eq_pool', clear
sort pid t_idx
qui count
di as result "  AB equations built: " r(N)

    *--------------------------------------------------------------*
    * 3. GMM in Mata (one-step and two-step, same formulas as the Python script)
    *--------------------------------------------------------------*
    mata:
        st_view(Dy=., ., "Dy")
        st_view(X=., ., ("Dy_lag","Dy_lag_x_hp"))
        st_view(Z=., ., ("z_lag2","z_lag3","z_lag2_hp","z_lag3_hp"))
        st_view(pid=., ., "pid")

        // One-step with W1 = (Z'Z)^-1
        ZX = Z'X
        Zy = Z'Dy
        W1 = invsym(Z'Z)
        XZWZX1 = ZX' * W1 * ZX
        beta1  = invsym(XZWZX1) * (ZX' * W1 * Zy)
        resid1 = Dy - X*beta1

        // Sum over children of the outer products of the scores, one-step residuals
        pids_u = uniqrows(pid)
        G = rows(pids_u)
        K = cols(Z)
        score_sum = J(K, K, 0)
        for (g=1; g<=G; g++) {
            idx = (pid :== pids_u[g])
            Zg  = select(Z, idx)
            ug  = select(resid1, idx)
            sg  = Zg' * ug
            score_sum = score_sum + sg * sg'
        }

        // One-step variance, clustered by child
        XZWZinv1 = invsym(XZWZX1)
        Var1 = XZWZinv1 * ZX' * W1 * score_sum * W1 * ZX * XZWZinv1 * (G/(G-1))
        se1  = sqrt(diagonal(Var1))

        // Two-step estimate with W2 = inverse of score_sum
        W2 = invsym(score_sum)
        XZWZX2 = ZX' * W2 * ZX
        beta2  = invsym(XZWZX2) * (ZX' * W2 * Zy)
        resid2 = Dy - X*beta2

        // Two-step variance: sandwich formula with the scores at the two-step residuals
        score_new = J(K, K, 0)
        for (g=1; g<=G; g++) {
            idx = (pid :== pids_u[g])
            Zg  = select(Z, idx)
            ug  = select(resid2, idx)
            sg  = Zg' * ug
            score_new = score_new + sg * sg'
        }
        XZWZinv2 = invsym(XZWZX2)
        Var2 = XZWZinv2 * ZX' * W2 * score_new * W2 * ZX * XZWZinv2 * (G/(G-1))
        se2  = sqrt(diagonal(Var2))

        // Hansen J with the two-step residuals and the two-step weight matrix
        g_sum = Z' * resid2
        Jstat = (g_sum' * W2 * g_sum)
        df_J = cols(Z) - cols(X)

        // Wave index of each equation. The test of second-order serial
        // correlation is computed in 05_method2_abgmm.py.
        st_view(t_idx=., ., "t_idx")

        printf("\n=== AB-GMM one-step (W1 = (Z'Z)^-1, cluster-robust SE on pid) ===\n")
        printf("  N eq = %g, G clusters = %g\n", rows(Dy), G)
        printf("  rho_0 (L.y_tilde_w)         = %9.4f   SE = %7.4f   z = %6.2f\n", ///
            beta1[1], se1[1], beta1[1]/se1[1])
        printf("  rho_1 (L.y_tilde_w x HP14)  = %9.4f   SE = %7.4f   z = %6.2f\n", ///
            beta1[2], se1[2], beta1[2]/se1[2])

        printf("\n=== AB-GMM two-step (W2 = (Sum Z_i u u' Z_i)^-1, cluster-robust SE) ===\n")
        printf("  rho_0 (L.y_tilde_w)         = %9.4f   SE = %7.4f   z = %6.2f\n", ///
            beta2[1], se2[1], beta2[1]/se2[1])
        printf("  rho_1 (L.y_tilde_w x HP14)  = %9.4f   SE = %7.4f   z = %6.2f\n", ///
            beta2[2], se2[2], beta2[2]/se2[2])
        printf("  Hansen J = %6.4f   (df = %g)\n", Jstat, df_J)

        // Store the estimates in Stata locals
        st_local("rho0_1s",   strofreal(beta1[1]))
        st_local("se_rho0_1s",strofreal(se1[1]))
        st_local("rho1_1s",   strofreal(beta1[2]))
        st_local("se_rho1_1s",strofreal(se1[2]))
        st_local("rho0_2s",   strofreal(beta2[1]))
        st_local("se_rho0_2s",strofreal(se2[1]))
        st_local("rho1_2s",   strofreal(beta2[2]))
        st_local("se_rho1_2s",strofreal(se2[2]))
        st_local("J_stat",    strofreal(Jstat))
        st_local("df_J",      strofreal(df_J))
        st_local("n_eq",      strofreal(rows(Dy)))
        st_local("n_clust",   strofreal(G))
    end

* Load the long panel again for xtabond2
use "$BASE_INT/panel_resid.dta", clear
drop if missing(y_tilde_w) | missing(HighParentOcc14)
destring pid year HighParentOcc14, replace force
xtset pid year, delta(2)

*----------------------------------------------------------*
* 4. xtabond2 for comparison
*    xtabond2 uses the H matrix of Arellano and Bond as the one-step
*    weight matrix and the call below includes year indicators, also
*    as instruments. Its point estimates therefore differ from the
*    Mata estimates above; the two are different specifications.
*----------------------------------------------------------*
gen double lag_y_x_hp = l.y_tilde_w * HighParentOcc14
di as result _n "=== Roodman-style xtabond2 (default H-matrix one-step) ==="
di as result "  (Reported for comparison with the Mata estimates above."
di as result "   The xtabond2 call includes year indicators.)"
xtabond2 y_tilde_w l.y_tilde_w lag_y_x_hp i.year,                ///
    gmm(l.y_tilde_w, lag(2 3) collapse)                          ///
    gmm(lag_y_x_hp, lag(2 3) collapse)                           ///
    iv(i.year) noleveleq twostep robust
estimates store AB_xtabond2

*----------------------------------------------------------*
* 5. Value of rho_1 implied by the minimum-distance variance shares
*----------------------------------------------------------*
preserve
    insheet using "$BASE_INT/method1_results.csv", clear comma
    keep if model == "FE"
    levelsof group, local(groups)
    foreach g of local groups {
        qui levelsof rho     if group == "`g'", local(rho_`g')   clean
        qui levelsof s_perm  if group == "`g'", local(s_`g')     clean
    }
restore

local rho_v_pooled = `rho_pooled'
local s_H = `s_H'
local s_L = `s_L'
local Delta_share = (1 - `rho_v_pooled') * (`s_H' - `s_L')
local excess      = `rho1_2s' - `Delta_share'
di as result _n "=== rho_1 implied by the minimum-distance variance shares ==="
di as result "  rho_v_pooled = " %5.4f `rho_v_pooled' ///
    "  s_H = " %5.4f `s_H' "  s_L = " %5.4f `s_L'
di as result "  Delta_share  = " %5.4f `Delta_share'
di as result "  rho_1_hat (2-step Mata) = " %5.4f `rho1_2s'
di as result "  excess (rho_1 - Delta_share) = " %5.4f `excess'

*----------------------------------------------------------*
* 6. Save the estimates
*----------------------------------------------------------*
tempname H
postfile `H' str20 estimator str20 coef double est double se double n_obs ///
    using "$BASE_INT/method2_results_stata.dta", replace
post `H' ("WG")        ("rho")         (`rho_wg')      (`se_wg')       (`n_wg')
post `H' ("AB1")       ("rho_0")       (`rho0_1s')     (`se_rho0_1s')  (`n_eq')
post `H' ("AB1")       ("rho_1")       (`rho1_1s')     (`se_rho1_1s')  (`n_eq')
post `H' ("AB2")       ("rho_0")       (`rho0_2s')     (`se_rho0_2s')  (`n_eq')
post `H' ("AB2")       ("rho_1")       (`rho1_2s')     (`se_rho1_2s')  (`n_eq')
post `H' ("AB2")       ("Hansen_J")    (`J_stat')      (.)             (`df_J')
post `H' ("benchmark") ("Delta_share") (`Delta_share') (.)             (.)
post `H' ("benchmark") ("excess")      (`excess')      (.)             (.)
postclose `H'
use "$BASE_INT/method2_results_stata.dta", clear

* The same table under the file name method2_results.dta
save "$BASE_INT/method2_results.dta", replace

di as result _n "=== 05_method2_abgmm.do: complete ==="
di "AB-GMM two-step (Mata, same formulas as the Python script)"
di "  rho_0 = `rho0_2s'  SE = `se_rho0_2s'"
di "  rho_1 = `rho1_2s'  SE = `se_rho1_2s'"
di "  Hansen J = `J_stat' (df = `df_J')"
