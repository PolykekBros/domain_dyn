use anyhow::Result;
use plotters::prelude::*;
use std::iter::Sum;
use std::ops::Add;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 800;

#[derive(Clone, Copy)]
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
        .vec_prod_const(m_abs)
        .normalize()
    }

    fn vec_prod(v1: Self, v2: Self) -> Self {
        Self {
            x: v1.y * v2.z - v1.z * v2.y,
            y: v1.x * v2.z - v1.z * v2.x,
            z: v1.x * v2.y - v1.y * v2.x,
        }
    }

    fn vec_prod_const(self, a: f64) -> Self {
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

fn plot_lat(lat: Vec<MagnMoment>, n: usize, m: usize, filename: &str) -> Result<()> {
    let root = BitMapBackend::new(filename, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Boolean Matrix Heatmap", ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..m, 0..n)?;
    chart.configure_mesh().draw()?;
    for element in lat.iter().enumerate().map(|(i, value)| {
        let y = i / m;
        let x = i % m;
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

fn gen_lattice(n: usize, m: usize) -> Vec<MagnMoment> {
    (0..n * m).map(|_| MagnMoment::random(1.0)).collect()
}

fn get_h_eff(
    lat: &[MagnMoment],
    n: usize,
    m: usize,
    coord: (usize, usize),
    j: f64,
    k: f64,
    h_ext: MagnMoment,
    l_axis: MagnMoment,
) -> MagnMoment {
    let h_exc = exchange_inter(lat, m, j, coord);
    let h_ani = aniso_inter(lat, m, k, coord, l_axis);
    let h_dipole = dipol_inter(lat, n, coord);
    h_ext + h_exc + h_ani + h_dipole
}

fn exchange_inter(
    lat: &[MagnMoment],
    m: usize,
    j: f64,
    (row, col): (usize, usize),
) -> MagnMoment {
    (lat[m * row + col]
        + lat[m * row + col - 1]
        + lat[m * row + col + 1]
        + lat[m * (row - 1) + col]
        + lat[m * (row + 1) + col])
        .vec_prod_const(j)
}

fn aniso_inter(
    lat: &[MagnMoment],
    m: usize,
    k: f64,
    (row, col): (usize, usize),
    l_axis: MagnMoment,
) -> MagnMoment {
    l_axis.vec_prod_const(2.0 * k * lat[m * row + col].scalar_prod(l_axis))
}

fn dipol_inter(lat: &[MagnMoment], m: usize, (row, col): (usize, usize)) -> MagnMoment {
    lat.iter()
        .enumerate()
        .filter(|(idx, _)| *idx != row * m + col)
        .map(|(idx, s)| {
            let row_j = (idx / m) as f64;
            let col_j = (idx % m) as f64;
            let r_ij = MagnMoment {
                x: (col as f64 - col_j),
                y: (row as f64 - row_j),
                z: 0.0,
            };
            let r_ij_len = r_ij.get_abs();
                s.vec_prod_const(1.0 / r_ij_len.powi(3))
                    + r_ij.vec_prod_const(-3.0 * s.scalar_prod(r_ij) / r_ij_len.powi(5))
        })
        .sum::<MagnMoment>()
}

fn main() {
    let n = 30;
    let m = 30;
    let h_ext = MagnMoment {
        x: 0.0,
        y: 0.0,
        z: 2.0,
    };
    let l_axis = MagnMoment {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    let lat = gen_lattice(100, 100);
    let h_eff = get_h_eff(&lat, n, m, (5, 5), 10.0, 48.0, h_ext, l_axis);
    println!("H_eff = {}", h_eff.get_abs());
    plot_lat(lat, n, m, "init_lat.png").unwrap();
}
