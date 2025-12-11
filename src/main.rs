use anyhow::Result;
use plotters::prelude::*;
use rayon::prelude::*;
use std::iter::Sum;
use std::ops::Add;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 800;

#[derive(Default, Clone, Copy)]
struct MagnMoment {
    x: f64,
    y: f64,
    z: f64,
}

impl Add for MagnMoment {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Add<&MagnMoment> for MagnMoment {
    type Output = Self;
    fn add(self, other: &MagnMoment) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sum for MagnMoment {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(
            Self {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            |acc, item| acc + item,
        )
    }
}

impl<'a> Sum<&'a MagnMoment> for MagnMoment {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a MagnMoment>,
    {
        iter.fold(
            MagnMoment {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            |acc, item| acc + item,
        )
    }
}

impl MagnMoment {
    fn normalize(self) -> Self {
        let abs = self.get_abs();
        Self {
            x: self.x / abs,
            y: self.y / abs,
            z: self.z / abs,
        }
    }

    fn random(m_abs: f64) -> Self {
        Self {
            x: rand::random(),
            y: rand::random(),
            z: rand::random(),
        }
        .const_prod(m_abs)
        .normalize()
    }

    fn const_prod(self, a: f64) -> Self {
        Self {
            x: self.x * a,
            y: self.y * a,
            z: self.z * a,
        }
    }

    fn scalar_prod(self, v1: Self) -> f64 {
        self.x * v1.x + self.y * v1.y + self.z * v1.z
    }

    fn get_abs(self) -> f64 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }
}

fn vec_prod(v1: MagnMoment, v2: MagnMoment) -> MagnMoment {
    MagnMoment {
        x: v1.y * v2.z - v1.z * v2.y,
        y: v1.x * v2.z - v1.z * v2.x,
        z: v1.x * v2.y - v1.y * v2.x,
    }
}

#[derive(Clone)]
struct Lattice {
    pub n: usize,
    pub m: usize,
    data: Vec<MagnMoment>,
}

impl Lattice {
    fn new(n: usize, m: usize) -> Self {
        Self {
            data: vec![MagnMoment::default(); n * m],
            n,
            m,
        }
    }

    fn random(n: usize, m: usize) -> Self {
        Self {
            data: (0..n * m).map(|_| MagnMoment::random(1.0)).collect(),
            n,
            m,
        }
    }

    fn get_index(&self, y: usize, x: usize) -> usize {
        debug_assert!(y < self.n);
        debug_assert!(x < self.m);
        y * self.m + x
    }

    fn get_position(&self, index: usize) -> (usize, usize) {
        debug_assert!(index < self.n * self.m);
        (index / self.m, index % self.m)
    }

    fn get_index_periodic(&self, y: i64, x: i64) -> usize {
        let n_i64 = self.n as i64;
        let m_i64 = self.m as i64;
        debug_assert!(y > -n_i64 && y < n_i64 * 2);
        debug_assert!(x > -m_i64 && x < m_i64 * 2);
        let y = ((y % n_i64 + n_i64) % n_i64) as usize;
        let x = ((x % m_i64 + m_i64) % m_i64) as usize;
        self.get_index(y, x)
    }

    fn get(&self, y: usize, x: usize) -> &MagnMoment {
        let index = self.get_index(y, x);
        &self.data[index]
    }

    fn get_mut(&mut self, y: usize, x: usize) -> &mut MagnMoment {
        let index = self.get_index(y, x);
        &mut self.data[index]
    }

    fn get_periodic(&self, y: i64, x: i64) -> &MagnMoment {
        let index = self.get_index_periodic(y, x);
        &self.data[index]
    }

    fn get_periodic_mut(&mut self, y: i64, x: i64) -> &mut MagnMoment {
        let index = self.get_index_periodic(y, x);
        &mut self.data[index]
    }

    fn iter(&self) -> std::slice::Iter<'_, MagnMoment> {
        self.data.iter()
    }

    fn par_iter(&self) -> rayon::slice::Iter<'_, MagnMoment> {
        self.data.par_iter()
    }

    fn iter_mut(&mut self) -> std::slice::IterMut<'_, MagnMoment> {
        self.data.iter_mut()
    }
}

impl IntoIterator for Lattice {
    type Item = MagnMoment;
    type IntoIter = std::vec::IntoIter<MagnMoment>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a> IntoIterator for &'a Lattice {
    type Item = &'a MagnMoment;
    type IntoIter = std::slice::Iter<'a, MagnMoment>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a> IntoIterator for &'a mut Lattice {
    type Item = &'a mut MagnMoment;
    type IntoIter = std::slice::IterMut<'a, MagnMoment>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

fn plot_lat(lat: &Lattice, filename: &str) -> Result<()> {
    let root = BitMapBackend::new(filename, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Boolean Matrix Heatmap", ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..lat.m, 0..lat.n)?;
    chart.configure_mesh().draw()?;
    for element in lat.iter().enumerate().map(|(i, value)| {
        let (y, x) = lat.get_position(i);
        let color_val = ((value.x + 1.0) / 2.0 * 255.0) as u8;
        let color = RGBColor(color_val, color_val, color_val);
        Rectangle::new([(x, y), (x + 1, y + 1)], color.filled())
    }) {
        chart.draw_series(std::iter::once(element))?;
    }
    root.present()?;
    println!("✅ Heatmap saved to {filename}");
    Ok(())
}

fn runge_kutta(
    lat: Lattice,
    gamma: f64,
    alpha: f64,
    dt: f64,
    j: f64,
    k: f64,
    h_ext: MagnMoment,
    l_axis: MagnMoment,
) -> Lattice {
    let data = lat
        .par_iter()
        .enumerate()
        .map(|(idx, magn)| {
            let coord = lat.get_position(idx);
            let h_eff = get_h_eff(&lat, coord, j, k, h_ext, l_axis);
            let k1 = llg_eq(gamma, alpha, *magn, h_eff).const_prod(dt);
            // h_eff = get_h_eff(&lat_prev, coord, j, k, h_ext, l_axis);
            let k2 = llg_eq(gamma, alpha, *magn + k1.const_prod(1.0 / 2.0), h_eff).const_prod(dt);
            // h_eff = get_h_eff(&lat_prev, coord, j, k, h_ext, l_axis);
            let k3 = llg_eq(gamma, alpha, *magn + k2.const_prod(1.0 / 2.0), h_eff).const_prod(dt);
            // h_eff = get_h_eff(&lat_prev, coord, j, k, h_ext, l_axis);
            let k4 = llg_eq(gamma, alpha, *magn + k3, h_eff).const_prod(dt);
            (lat.data[idx]
                + (k1 + k2.const_prod(2.0) + k3.const_prod(2.0) + k4).const_prod(1.0 / 6.0))
            .normalize()
        })
        .collect();
    Lattice {
        n: lat.n,
        m: lat.m,
        data,
    }
}

fn llg_eq(gamma: f64, alpha: f64, magn: MagnMoment, h_eff: MagnMoment) -> MagnMoment {
    (vec_prod(magn, h_eff)
        + vec_prod(magn, vec_prod(magn, h_eff)).const_prod(alpha / magn.get_abs()))
    .const_prod(-gamma / (1.0 + alpha.powi(2)))
}

fn get_h_eff(
    lat: &Lattice,
    coord: (usize, usize),
    j: f64,
    k: f64,
    h_ext: MagnMoment,
    l_axis: MagnMoment,
) -> MagnMoment {
    let h_exc = exchange_inter(lat, j, coord);
    let h_ani = aniso_inter(lat, k, coord, l_axis);
    let h_dipole = dipol_inter(lat, coord);
    h_ext + h_exc + h_ani + h_dipole
}

fn exchange_inter(lat: &Lattice, j: f64, (row, col): (usize, usize)) -> MagnMoment {
    let row = row as i64;
    let col = col as i64;
    (*lat.get_periodic(row, col)
        + *lat.get_periodic(row, col - 1)
        + *lat.get_periodic(row, col + 1)
        + *lat.get_periodic(row - 1, col)
        + *lat.get_periodic(row + 1, col))
    .const_prod(j)
}

fn aniso_inter(
    lat: &Lattice,
    k: f64,
    (row, col): (usize, usize),
    l_axis: MagnMoment,
) -> MagnMoment {
    l_axis.const_prod(2.0 * k * lat.get(row, col).scalar_prod(l_axis))
}

fn dipol_inter(lat: &Lattice, (row, col): (usize, usize)) -> MagnMoment {
    let half_n = ((lat.n - 1) / 2) as i64;
    let half_m = ((lat.m - 1) / 2) as i64;
    (-half_n..=half_n)
        .flat_map(|d_n| (-half_m..=half_m).map(move |d_m| (d_n, d_m)))
        .filter(|(d_n, d_m)| *d_n != 0 && *d_m != 0)
        .map(|(d_n, d_m)| {
            let r_ij = MagnMoment {
                x: d_n as f64,
                y: d_m as f64,
                z: 0.0,
            };
            let r_ij_len = r_ij.get_abs();
            let row_j = row as i64 + d_n;
            let col_j = col as i64 + d_m;
            let s = lat.get_periodic(row_j, col_j);
            s.const_prod(1.0 / r_ij_len.powi(3))
                + r_ij.const_prod(-3.0 * s.scalar_prod(r_ij) / r_ij_len.powi(5))
        })
        .sum::<MagnMoment>()
}

fn main() {
    let j = 21.0 * 10.0_f64.powi(-12);
    let k = 4.8 * 10.0_f64.powi(4);
    let gamma = 1.76_f64 * 10.0_f64.powi(11);
    let alpha = 0.01;

    let time = 500;
    let dt = 20.0_f64.powi(-9);
    let n = 30;
    let m = 30;
    let h_ext = MagnMoment {
        x: 0.0,
        y: 0.0,
        z: 10.0,
    };
    let l_axis = MagnMoment {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    let mut lat = Lattice::random(n, m);
    let h_eff = get_h_eff(&lat, (5, 5), j, k, h_ext, l_axis);
    println!("H_eff = {}", h_eff.get_abs());
    plot_lat(&lat, "init_lat.png").unwrap();
    for t in 0..time {
        lat = runge_kutta(lat, gamma, alpha, dt, j, k, h_ext, l_axis);
        plot_lat(&lat, &format!("anim/{t}.png")).unwrap();
    }
    plot_lat(&lat, "final_lat.png").unwrap();
}
