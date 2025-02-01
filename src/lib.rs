use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use ordered_float::OrderedFloat as F;
use pyo3::prelude::*;
use rangemap::RangeMap;
use skymask_rs::data::read_shp;
use skymask_rs::utils::{ProjLine, ProjSegment};

#[pyclass]
struct World {
    lines: Array2<f64>,
    kdtree: KdTree<f64, usize, [f64; 2]>,
    #[pyo3(get, set)]
    max_dist: f64,
    #[pyo3(get, set)]
    eps: f64,
}

impl World {
    fn calc_skymask(&self, pos: [f64; 2]) -> RangeMap<F<f64>, (F<f64>, F<f64>)> {
        let lines_iter = self
            .kdtree
            .within(&pos, self.max_dist.powi(2), &squared_euclidean)
            .unwrap()
            .into_iter()
            .filter_map(|(_, &i)| {
                let row = self.lines.row(i);
                ProjSegment::<F<f64>, (F<f64>, F<f64>)>::from_points(
                    &[F(row[0] - pos[0]), F(row[1] - pos[1]), F(row[2])],
                    &[F(row[3] - pos[0]), F(row[4] - pos[1]), F(row[5])],
                )
            });
        skymask_rs::skymask(lines_iter, F(self.eps))
    }
}

#[pymethods]
impl World {
    #[new]
    fn new(path: &str, max_dist: f64, eps: f64) -> Self {
        let (lines, _, kdtree) = read_shp(path);
        Self {
            lines: lines,
            kdtree: kdtree,
            max_dist,
            eps,
        }
    }

    fn skymask(&self, pos: [f64; 2]) -> Vec<((f64, f64), (f64, f64))> {
        let func = self.calc_skymask(pos);
        func.into_iter()
            .map(|(x, (y1, y2))| ((x.start.0, x.end.0), (y1.0, y2.0)))
            .collect()
    }

    fn skymask_samples<'py>(
        &self,
        py: Python<'py>,
        pos: [f64; 2],
        samples: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let func = self.calc_skymask(pos);
        let res = Array1::from_iter(
            samples
                .as_array()
                .iter()
                .map(|&x| func.get(&F(x)).map(|f| f.at(F(x)).0).unwrap_or(0.0)),
        );
        res.into_pyarray(py)
    }

    fn skymask_par_samples<'py>(
        &self,
        py: Python<'py>,
        pos: [f64; 2],
        samples: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        use rayon::prelude::*;
        let func = self.calc_skymask(pos);
        let res = Array1::from_iter(
            samples
                .as_slice()
                .unwrap()
                .par_iter()
                .map(|&x| func.get(&F(x)).map(|f| f.at(F(x)).0).unwrap_or(0.0))
                .collect::<Vec<_>>(),
        );
        res.into_pyarray(py)
    }
}

#[pymodule]
fn skymask_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<World>()?;
    Ok(())
}
