from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FJWMMAResult:
    xmma: np.ndarray
    ymma: np.ndarray
    zmma: float
    lam: np.ndarray
    xsi: np.ndarray
    eta: np.ndarray
    mu: np.ndarray
    zet: float
    s: np.ndarray
    low: np.ndarray
    upp: np.ndarray
    alfa: np.ndarray
    beta: np.ndarray


def _column(values: np.ndarray | list[float] | tuple[float, ...], *, size: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != size:
        raise ValueError(f"{label} size mismatch: {array.size} != {size}.")
    return array


def _matrix(values: np.ndarray | list[list[float]], *, shape: tuple[int, int], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{label} shape mismatch: {array.shape} != {shape}.")
    return array


def mmasub(
    *,
    m: int,
    n: int,
    iteration: int,
    xval: np.ndarray,
    xmin: np.ndarray,
    xmax: np.ndarray,
    f0val: float,
    df0dx: np.ndarray,
    fval: np.ndarray,
    dfdx: np.ndarray,
    xold1: np.ndarray,
    xold2: np.ndarray,
    low: np.ndarray,
    upp: np.ndarray,
    a0: float,
    a: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> FJWMMAResult:
    """Port of Svanberg's September 2007 MMA `mmasub.m`.

    The historical FJW workflow calls this routine with one volume constraint.
    The implementation keeps the original dense algebra and avoids external
    optimizer dependencies, so the Python update path can be compared directly
    against `references/fjw_work/mmasub.m` and `subsolv.m`.
    """

    del f0val
    m = int(m)
    n = int(n)
    if m <= 0 or n <= 0:
        raise ValueError("m and n must be positive.")

    xval = _column(xval, size=n, label="xval")
    xmin = _column(xmin, size=n, label="xmin")
    xmax = _column(xmax, size=n, label="xmax")
    xold1 = _column(xold1, size=n, label="xold1")
    xold2 = _column(xold2, size=n, label="xold2")
    low = _column(low, size=n, label="low")
    upp = _column(upp, size=n, label="upp")
    df0dx = _column(df0dx, size=n, label="df0dx")
    fval = _column(fval, size=m, label="fval")
    dfdx = _matrix(dfdx, shape=(m, n), label="dfdx")
    a = _column(a, size=m, label="a")
    c = _column(c, size=m, label="c")
    d = _column(d, size=m, label="d")

    epsimin = 1.0e-7
    raa0 = 1.0e-5
    albefa = 0.1
    asyinit = 0.5
    asyincr = 1.2
    asydecr = 0.7
    eeen = np.ones(n, dtype=np.float64)
    eeem = np.ones(m, dtype=np.float64)

    if iteration < 2.5:
        low = xval - asyinit * (xmax - xmin)
        upp = xval + asyinit * (xmax - xmin)
    else:
        zzz = (xval - xold1) * (xold1 - xold2)
        factor = eeen.copy()
        factor[zzz > 0.0] = asyincr
        factor[zzz < 0.0] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        lowmin = xval - 10.0 * (xmax - xmin)
        lowmax = xval - 0.01 * (xmax - xmin)
        uppmin = xval + 0.01 * (xmax - xmin)
        uppmax = xval + 10.0 * (xmax - xmin)
        low = np.minimum(np.maximum(low, lowmin), lowmax)
        upp = np.maximum(np.minimum(upp, uppmax), uppmin)

    alfa = np.maximum(low + albefa * (xval - low), xmin)
    beta = np.minimum(upp - albefa * (upp - xval), xmax)

    xmami = np.maximum(xmax - xmin, 1.0e-5 * eeen)
    xmamiinv = eeen / xmami
    ux1 = upp - xval
    xl1 = xval - low
    ux2 = ux1 * ux1
    xl2 = xl1 * xl1
    uxinv = eeen / ux1
    xlinv = eeen / xl1

    p0 = np.maximum(df0dx, 0.0)
    q0 = np.maximum(-df0dx, 0.0)
    pq0 = 0.001 * (p0 + q0) + raa0 * xmamiinv
    p0 = (p0 + pq0) * ux2
    q0 = (q0 + pq0) * xl2

    p_mat = np.maximum(dfdx, 0.0)
    q_mat = np.maximum(-dfdx, 0.0)
    pq_mat = 0.001 * (p_mat + q_mat) + raa0 * np.outer(eeem, xmamiinv)
    p_mat = (p_mat + pq_mat) * ux2[None, :]
    q_mat = (q_mat + pq_mat) * xl2[None, :]
    b = p_mat @ uxinv + q_mat @ xlinv - fval

    sub = subsolv(
        m=m,
        n=n,
        epsimin=epsimin,
        low=low,
        upp=upp,
        alfa=alfa,
        beta=beta,
        p0=p0,
        q0=q0,
        p_mat=p_mat,
        q_mat=q_mat,
        a0=float(a0),
        a=a,
        b=b,
        c=c,
        d=d,
    )
    return FJWMMAResult(
        xmma=sub.xmma,
        ymma=sub.ymma,
        zmma=sub.zmma,
        lam=sub.lam,
        xsi=sub.xsi,
        eta=sub.eta,
        mu=sub.mu,
        zet=sub.zet,
        s=sub.s,
        low=low,
        upp=upp,
        alfa=alfa,
        beta=beta,
    )


def subsolv(
    *,
    m: int,
    n: int,
    epsimin: float,
    low: np.ndarray,
    upp: np.ndarray,
    alfa: np.ndarray,
    beta: np.ndarray,
    p0: np.ndarray,
    q0: np.ndarray,
    p_mat: np.ndarray,
    q_mat: np.ndarray,
    a0: float,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> FJWMMAResult:
    een = np.ones(n, dtype=np.float64)
    eem = np.ones(m, dtype=np.float64)

    epsi = 1.0
    x = 0.5 * (alfa + beta)
    y = eem.copy()
    z = 1.0
    lam = eem.copy()
    xsi = np.maximum(een / (x - alfa), een)
    eta = np.maximum(een / (beta - x), een)
    mu = np.maximum(eem, 0.5 * c)
    zet = 1.0
    s = eem.copy()

    while epsi > epsimin:
        epsvecn = epsi * een
        epsvecm = epsi * eem

        residunorm, residumax = _subsolv_residual(
            x=x,
            y=y,
            z=z,
            lam=lam,
            xsi=xsi,
            eta=eta,
            mu=mu,
            zet=zet,
            s=s,
            low=low,
            upp=upp,
            alfa=alfa,
            beta=beta,
            p0=p0,
            q0=q0,
            p_mat=p_mat,
            q_mat=q_mat,
            a0=a0,
            a=a,
            b=b,
            c=c,
            d=d,
            epsvecn=epsvecn,
            epsvecm=epsvecm,
        )
        ittt = 0
        resinew = residunorm
        while residumax > 0.9 * epsi and ittt < 200:
            ittt += 1
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2
            uxinv1 = een / ux1
            xlinv1 = een / xl1
            uxinv2 = een / ux2
            xlinv2 = een / xl2

            plam = p0 + p_mat.T @ lam
            qlam = q0 + q_mat.T @ lam
            gvec = p_mat @ uxinv1 + q_mat @ xlinv1
            gg = p_mat * uxinv2[None, :] - q_mat * xlinv2[None, :]
            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x)
            dely = c + d * y - lam - epsvecm / y
            delz = a0 - float(a @ lam) - epsi / z
            dellam = gvec - a * z - y - b + epsvecm / lam

            diagx = 2.0 * (plam / ux3 + qlam / xl3) + xsi / (x - alfa) + eta / (beta - x)
            diagxinv = een / diagx
            diagy = d + mu / y
            diagyinv = eem / diagy
            diaglam = s / lam
            diaglamyi = diaglam + diagyinv

            if m < n:
                blam = dellam + dely / diagy - gg @ (delx / diagx)
                bb = np.concatenate((blam, np.array([delz], dtype=np.float64)))
                alam = np.diag(diaglamyi) + (gg * diagxinv[None, :]) @ gg.T
                aa = np.empty((m + 1, m + 1), dtype=np.float64)
                aa[:m, :m] = alam
                aa[:m, m] = a
                aa[m, :m] = a
                aa[m, m] = -zet / z
                solut = np.linalg.solve(aa, bb)
                dlam = solut[:m]
                dz = float(solut[m])
                dx = -delx / diagx - (gg.T @ dlam) / diagx
            else:
                diaglamyiinv = eem / diaglamyi
                dellamyi = dellam + dely / diagy
                axx = np.diag(diagx) + (gg.T * diaglamyiinv[None, :]) @ gg
                azz = zet / z + float(a @ (a / diaglamyi))
                axz = -(gg.T @ (a / diaglamyi))
                bx = delx + gg.T @ (dellamyi / diaglamyi)
                bz = delz - float(a @ (dellamyi / diaglamyi))
                aa = np.empty((n + 1, n + 1), dtype=np.float64)
                aa[:n, :n] = axx
                aa[:n, n] = axz
                aa[n, :n] = axz
                aa[n, n] = azz
                bb = np.concatenate((-bx, np.array([-bz], dtype=np.float64)))
                solut = np.linalg.solve(aa, bb)
                dx = solut[:n]
                dz = float(solut[n])
                dlam = (gg @ dx) / diaglamyi - dz * (a / diaglamyi) + dellamyi / diaglamyi

            dy = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsvecn / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = -eta + epsvecn / (beta - x) + (eta * dx) / (beta - x)
            dmu = -mu + epsvecm / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsvecm / lam - (s * dlam) / lam

            xx = np.concatenate((y, np.array([z]), lam, xsi, eta, mu, np.array([zet]), s))
            dxx = np.concatenate((dy, np.array([dz]), dlam, dxsi, deta, dmu, np.array([dzet]), ds))
            stepxx = np.max(-1.01 * dxx / xx)
            stepalfa = np.max(-1.01 * dx / (x - alfa))
            stepbeta = np.max(1.01 * dx / (beta - x))
            steg = 1.0 / max(stepxx, stepalfa, stepbeta, 1.0)

            xold = x.copy()
            yold = y.copy()
            zold = float(z)
            lamold = lam.copy()
            xsiold = xsi.copy()
            etaold = eta.copy()
            muold = mu.copy()
            zetold = float(zet)
            sold = s.copy()

            itto = 0
            resinew = 2.0 * residunorm
            while resinew > residunorm and itto < 50:
                itto += 1
                x = xold + steg * dx
                y = yold + steg * dy
                z = zold + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu = muold + steg * dmu
                zet = zetold + steg * dzet
                s = sold + steg * ds
                resinew, _ = _subsolv_residual(
                    x=x,
                    y=y,
                    z=z,
                    lam=lam,
                    xsi=xsi,
                    eta=eta,
                    mu=mu,
                    zet=zet,
                    s=s,
                    low=low,
                    upp=upp,
                    alfa=alfa,
                    beta=beta,
                    p0=p0,
                    q0=q0,
                    p_mat=p_mat,
                    q_mat=q_mat,
                    a0=a0,
                    a=a,
                    b=b,
                    c=c,
                    d=d,
                    epsvecn=epsvecn,
                    epsvecm=epsvecm,
                )
                steg *= 0.5

            residunorm = resinew
            _, residumax = _subsolv_residual(
                x=x,
                y=y,
                z=z,
                lam=lam,
                xsi=xsi,
                eta=eta,
                mu=mu,
                zet=zet,
                s=s,
                low=low,
                upp=upp,
                alfa=alfa,
                beta=beta,
                p0=p0,
                q0=q0,
                p_mat=p_mat,
                q_mat=q_mat,
                a0=a0,
                a=a,
                b=b,
                c=c,
                d=d,
                epsvecn=epsvecn,
                epsvecm=epsvecm,
            )

        epsi *= 0.1

    return FJWMMAResult(
        xmma=x,
        ymma=y,
        zmma=float(z),
        lam=lam,
        xsi=xsi,
        eta=eta,
        mu=mu,
        zet=float(zet),
        s=s,
        low=low,
        upp=upp,
        alfa=alfa,
        beta=beta,
    )


def _subsolv_residual(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: float,
    lam: np.ndarray,
    xsi: np.ndarray,
    eta: np.ndarray,
    mu: np.ndarray,
    zet: float,
    s: np.ndarray,
    low: np.ndarray,
    upp: np.ndarray,
    alfa: np.ndarray,
    beta: np.ndarray,
    p0: np.ndarray,
    q0: np.ndarray,
    p_mat: np.ndarray,
    q_mat: np.ndarray,
    a0: float,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    epsvecn: np.ndarray,
    epsvecm: np.ndarray,
) -> tuple[float, float]:
    ux1 = upp - x
    xl1 = x - low
    ux2 = ux1 * ux1
    xl2 = xl1 * xl1
    uxinv1 = 1.0 / ux1
    xlinv1 = 1.0 / xl1
    plam = p0 + p_mat.T @ lam
    qlam = q0 + q_mat.T @ lam
    gvec = p_mat @ uxinv1 + q_mat @ xlinv1
    dpsidx = plam / ux2 - qlam / xl2

    rex = dpsidx - xsi + eta
    rey = c + d * y - mu - lam
    rez = np.array([a0 - zet - float(a @ lam)], dtype=np.float64)
    relam = gvec - a * z - y + s - b
    rexsi = xsi * (x - alfa) - epsvecn
    reeta = eta * (beta - x) - epsvecn
    remu = mu * y - epsvecm
    rezet = np.array([zet * z - float(epsvecm[0])], dtype=np.float64)
    res = lam * s - epsvecm
    residual = np.concatenate((rex, rey, rez, relam, rexsi, reeta, remu, rezet, res))
    return float(np.linalg.norm(residual)), float(np.max(np.abs(residual)))


__all__ = [
    "FJWMMAResult",
    "mmasub",
    "subsolv",
]
