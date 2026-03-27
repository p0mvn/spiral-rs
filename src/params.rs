use std::mem::size_of;

use crate::{
    arith::*,
    client::{MAX_EXP_DIM, SEED_LENGTH},
    ntt::*,
    number_theory::*,
    poly::*,
};

pub const MAX_MODULI: usize = 4;

pub static MIN_Q2_BITS: u64 = 14;
pub static Q2_VALUES: [u64; 37] = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    12289,
    12289,
    61441,
    65537,
    65537,
    520193,
    786433,
    786433,
    3604481,
    7340033,
    16515073,
    33292289,
    67043329,
    132120577,
    268369921,
    469762049,
    1073479681,
    2013265921,
    4293918721,
    8588886017,
    17175674881,
    34359214081,
    68718428161,
];

#[derive(Debug, PartialEq, Clone)]
pub struct Params {
    pub poly_len: usize,
    pub poly_len_log2: usize,
    pub ntt_tables: Vec<Vec<Vec<u64>>>,
    pub scratch: Vec<u64>,

    pub crt_count: usize,
    pub barrett_cr_0: [u64; MAX_MODULI],
    pub barrett_cr_1: [u64; MAX_MODULI],
    pub barrett_cr_0_modulus: u64,
    pub barrett_cr_1_modulus: u64,
    pub mod0_inv_mod1: u64,
    pub mod1_inv_mod0: u64,
    pub moduli: [u64; MAX_MODULI],
    pub modulus: u64,
    pub modulus_log2: u64,
    pub noise_width: f64,

    pub n: usize,
    pub pt_modulus: u64,
    pub q2_bits: u64,
    pub t_conv: usize,
    pub t_exp_left: usize,
    pub t_exp_right: usize,
    pub t_gsw: usize,

    pub expand_queries: bool,
    pub db_dim_1: usize,
    pub db_dim_2: usize,
    pub instances: usize,
    pub db_item_size: usize,

    pub version: usize,
}

impl Params {
    pub fn get_ntt_forward_table(&self, i: usize) -> &[u64] {
        self.ntt_tables[i][0].as_slice()
    }
    pub fn get_ntt_forward_prime_table(&self, i: usize) -> &[u64] {
        self.ntt_tables[i][1].as_slice()
    }
    pub fn get_ntt_inverse_table(&self, i: usize) -> &[u64] {
        self.ntt_tables[i][2].as_slice()
    }
    pub fn get_ntt_inverse_prime_table(&self, i: usize) -> &[u64] {
        self.ntt_tables[i][3].as_slice()
    }

    pub fn get_v_neg1(&self) -> Vec<PolyMatrixNTT> {
        let mut v_neg1 = Vec::new();
        for i in 0..self.poly_len_log2 {
            let idx = self.poly_len - (1 << i);
            let mut ng1 = PolyMatrixRaw::zero(&self, 1, 1);
            ng1.data[idx] = 1;
            v_neg1.push((-&ng1).ntt());
        }
        v_neg1
    }

    pub fn get_sk_gsw(&self) -> (usize, usize) {
        (self.n, 1)
    }
    pub fn get_sk_reg(&self) -> (usize, usize) {
        (1, 1)
    }

    pub fn num_expanded(&self) -> usize {
        1 << self.db_dim_1
    }

    pub fn num_items(&self) -> usize {
        (1 << self.db_dim_1) * (1 << self.db_dim_2)
    }

    pub fn item_size(&self) -> usize {
        let logp = log2(self.pt_modulus) as usize;
        self.instances * self.n * self.n * self.poly_len * logp / 8
    }

    pub fn g(&self) -> usize {
        let num_bits_to_gen = self.t_gsw * self.db_dim_2 + self.num_expanded();
        log2_ceil_usize(num_bits_to_gen)
    }

    pub fn stop_round(&self) -> usize {
        log2_ceil_usize(self.t_gsw * self.db_dim_2)
    }

    pub fn factor_on_first_dim(&self) -> usize {
        if self.db_dim_2 == 0 {
            1
        } else {
            2
        }
    }

    pub fn setup_bytes(&self) -> usize {
        let mut sz_polys = 0;

        let num_packing_mats = if self.version == 0 { self.n } else { 2 };
        let packing_sz = ((self.n + 1) - 1) * self.t_conv;
        sz_polys += num_packing_mats * packing_sz;

        if self.expand_queries {
            let expansion_left_sz = self.g().min(MAX_EXP_DIM) * self.t_exp_left;
            let mut expansion_right_sz = (self.stop_round() + 1) * self.t_exp_right;
            let conversion_sz = 2 * self.t_conv;

            if self.version > 0 && self.t_exp_left == self.t_exp_right {
                expansion_right_sz = 0;
            }

            sz_polys += expansion_left_sz + expansion_right_sz + conversion_sz;
        }

        let sz_bytes = sz_polys * self.poly_len * size_of::<u64>();
        SEED_LENGTH + sz_bytes
    }

    pub fn query_bytes(&self) -> usize {
        let sz_polys;
        let num_query_cts = if self.db_dim_1 > MAX_EXP_DIM {
            1 << (self.db_dim_1 - MAX_EXP_DIM)
        } else {
            1
        };

        if self.expand_queries {
            sz_polys = num_query_cts;
        } else {
            sz_polys = self.db_dim_1 * 2 * self.t_gsw;
        }

        let sz_bytes = sz_polys * self.poly_len * size_of::<u64>();
        SEED_LENGTH + sz_bytes
    }

    pub fn query_v_buf_bytes(&self) -> usize {
        self.num_expanded() * self.poly_len * size_of::<u64>()
    }

    pub fn bytes_per_chunk(&self) -> usize {
        let trials = self.n * self.n;
        let chunks = self.instances * trials;
        let bytes_per_chunk = f64::ceil(self.db_item_size as f64 / chunks as f64) as usize;
        bytes_per_chunk
    }

    pub fn modp_words_per_chunk(&self) -> usize {
        let bytes_per_chunk = self.bytes_per_chunk();
        let logp = log2(self.pt_modulus);
        let modp_words_per_chunk = f64::ceil((bytes_per_chunk * 8) as f64 / logp as f64) as usize;
        modp_words_per_chunk
    }

    pub fn crt_compose_1(&self, x: u64) -> u64 {
        assert_eq!(self.crt_count, 1);
        x
    }

    pub fn crt_compose_2(&self, x: u64, y: u64) -> u64 {
        assert_eq!(self.crt_count, 2);
        // assert!(self.moduli[0] > self.moduli[1]);
        //            n                 m

        let mut val = (x as u128) * (self.mod1_inv_mod0 as u128);
        val += (y as u128) * (self.mod0_inv_mod1 as u128);

        // let mut val = y as u128;
        // val += self.mod1_inv_mod0 as u128 * (x + self.moduli[0] - y) as u128;

        barrett_reduction_u128(self, val)
    }

    pub fn crt_compose(&self, a: &[u64], idx: usize) -> u64 {
        if self.crt_count == 1 {
            self.crt_compose_1(a[idx])
        } else {
            self.crt_compose_2(a[idx], a[idx + self.poly_len])
        }
    }

    pub fn init(
        poly_len: usize,
        moduli: &[u64],
        noise_width: f64,
        n: usize,
        pt_modulus: u64,
        q2_bits: u64,
        t_conv: usize,
        t_exp_left: usize,
        t_exp_right: usize,
        t_gsw: usize,
        expand_queries: bool,
        db_dim_1: usize,
        db_dim_2: usize,
        instances: usize,
        db_item_size: usize,
        version: usize,
    ) -> Self {
        assert!(q2_bits >= MIN_Q2_BITS);

        let poly_len_log2 = log2(poly_len as u64) as usize;
        let crt_count = moduli.len();
        assert!(crt_count <= MAX_MODULI);
        let mut moduli_array = [0; MAX_MODULI];
        for i in 0..crt_count {
            moduli_array[i] = moduli[i];
        }
        let ntt_tables = if crt_count > 1 {
            build_ntt_tables(poly_len, moduli, None)
        } else {
            build_ntt_tables_alt(poly_len, moduli, None)
        };
        let scratch = vec![0u64; crt_count * poly_len];
        let mut modulus = 1;
        for m in moduli {
            modulus *= m;
        }
        let modulus_log2 = log2_ceil(modulus);
        let (barrett_cr_0, barrett_cr_1) = get_barrett(moduli);
        let (barrett_cr_0_modulus, barrett_cr_1_modulus) = get_barrett_crs(modulus);
        let mut mod0_inv_mod1 = 0;
        let mut mod1_inv_mod0 = 0;
        if crt_count == 2 {
            mod0_inv_mod1 = moduli[0] * invert_uint_mod(moduli[0], moduli[1]).unwrap();
            mod1_inv_mod0 = moduli[1] * invert_uint_mod(moduli[1], moduli[0]).unwrap();
        }
        Self {
            poly_len,
            poly_len_log2,
            ntt_tables,
            scratch,
            crt_count,
            barrett_cr_0,
            barrett_cr_1,
            barrett_cr_0_modulus,
            barrett_cr_1_modulus,
            mod0_inv_mod1,
            mod1_inv_mod0,
            moduli: moduli_array,
            modulus,
            modulus_log2,
            noise_width,
            n,
            pt_modulus,
            q2_bits,
            t_conv,
            t_exp_left,
            t_exp_right,
            t_gsw,
            expand_queries,
            db_dim_1,
            db_dim_2,
            instances,
            db_item_size,
            version,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::util::*;

    fn default_moduli() -> Vec<u64> {
        vec![268369921u64, 249561089u64]
    }

    fn single_modulus() -> Vec<u64> {
        vec![180143985094819841u64]
    }

    /// Helper: build Params with standard defaults, overriding only the fields under test.
    fn make_params(
        moduli: &[u64],
        n: usize,
        pt_modulus: u64,
        q2_bits: u64,
        db_dim_1: usize,
        db_dim_2: usize,
        instances: usize,
    ) -> Params {
        Params::init(
            2048, moduli, 6.4, n, pt_modulus, q2_bits, 4, 8, 56, 8, true, db_dim_1, db_dim_2,
            instances, 2048, 0,
        )
    }

    // ----------------------------------------------------------------
    // q2_bits boundary tests
    // ----------------------------------------------------------------

    #[test]
    fn q2_bits_at_minimum_succeeds() {
        let params = make_params(&default_moduli(), 2, 256, MIN_Q2_BITS, 9, 6, 1);
        assert_eq!(params.q2_bits, MIN_Q2_BITS);
        assert!(Q2_VALUES[params.q2_bits as usize] > 0);
    }

    #[test]
    #[should_panic]
    fn q2_bits_below_minimum_panics() {
        make_params(&default_moduli(), 2, 256, MIN_Q2_BITS - 1, 9, 6, 1);
    }

    #[test]
    fn q2_bits_at_table_maximum_succeeds() {
        let max_q2 = (Q2_VALUES.len() - 1) as u64;
        let params = make_params(&default_moduli(), 2, 256, max_q2, 9, 6, 1);
        assert_eq!(params.q2_bits, max_q2);
        assert!(Q2_VALUES[params.q2_bits as usize] > 0);
    }

    // ----------------------------------------------------------------
    // Moduli count boundary tests
    // ----------------------------------------------------------------

    #[test]
    #[should_panic]
    fn too_many_moduli_panics() {
        let moduli = vec![268369921u64; MAX_MODULI + 1];
        Params::init(
            2048, &moduli, 6.4, 2, 256, 20, 4, 8, 56, 8, true, 9, 6, 1, 2048, 0,
        );
    }

    #[test]
    fn single_modulus_at_max_slot_succeeds() {
        let moduli = vec![268369921u64];
        let params = Params::init(
            2048, &moduli, 6.4, 2, 256, 20, 4, 8, 56, 8, true, 9, 6, 1, 2048, 0,
        );
        assert_eq!(params.crt_count, 1);
    }

    // ----------------------------------------------------------------
    // Single-modulus vs dual-modulus NTT path
    // ----------------------------------------------------------------

    #[test]
    fn single_modulus_params_uses_crt_count_1() {
        let params = make_params(&single_modulus(), 2, 256, 20, 9, 6, 1);
        assert_eq!(params.crt_count, 1);
        assert_eq!(params.modulus, single_modulus()[0]);
        assert_eq!(params.ntt_tables.len(), 1);
    }

    #[test]
    fn dual_modulus_params_uses_crt_count_2() {
        let params = make_params(&default_moduli(), 2, 256, 20, 9, 6, 1);
        assert_eq!(params.crt_count, 2);
        assert_eq!(params.modulus, default_moduli()[0] * default_moduli()[1]);
        assert_eq!(params.ntt_tables.len(), 2);
        assert_ne!(params.mod0_inv_mod1, 0);
        assert_ne!(params.mod1_inv_mod0, 0);
    }

    #[test]
    fn single_modulus_crt_compose_is_identity() {
        let params = make_params(&single_modulus(), 2, 256, 20, 9, 6, 1);
        assert_eq!(params.crt_compose_1(42), 42);
        assert_eq!(params.crt_compose_1(0), 0);
        assert_eq!(params.crt_compose_1(params.modulus - 1), params.modulus - 1);
    }

    // ----------------------------------------------------------------
    // Derived parameter consistency
    // ----------------------------------------------------------------

    #[test]
    fn derived_values_consistent_for_test_params() {
        let params = get_test_params();
        assert_eq!(params.poly_len, 2048);
        assert_eq!(params.poly_len_log2, 11);
        assert_eq!(params.crt_count, 2);
        assert!(params.modulus > 0);
        assert_eq!(params.modulus, params.moduli[0] * params.moduli[1]);
        assert!(params.modulus_log2 > 0);
        assert!(params.modulus_log2 <= 64);

        assert!(params.ntt_tables.len() == params.crt_count);
        for table in &params.ntt_tables {
            assert_eq!(table.len(), 4);
            for subtable in table {
                assert_eq!(subtable.len(), params.poly_len);
            }
        }
    }

    #[test]
    fn derived_values_consistent_for_short_keygen_params() {
        let params = get_short_keygen_params();
        assert_eq!(params.poly_len, 2048);
        assert_eq!(params.n, 2);
        assert_eq!(params.pt_modulus, 256);
        assert_eq!(params.num_items(), (1 << 9) * (1 << 6));
        assert_eq!(params.num_expanded(), 1 << 9);
    }

    #[test]
    fn derived_values_consistent_for_no_expansion_params() {
        let params = get_no_expansion_testing_params();
        assert!(!params.expand_queries);
        assert_eq!(params.n, 5);
        assert_eq!(params.pt_modulus, 65536);
        assert!(params.q2_bits >= MIN_Q2_BITS);
    }

    // ----------------------------------------------------------------
    // pt_modulus boundary values
    // ----------------------------------------------------------------

    #[test]
    fn pt_modulus_power_of_two_values() {
        for p in [2u64, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 65536] {
            let params = make_params(&default_moduli(), 2, p, 20, 6, 2, 1);
            assert_eq!(params.pt_modulus, p);
            let q1 = 4 * p;
            assert!(log2_ceil(q1) <= 64, "q1 = 4*p must fit in u64 for p={}", p);
        }
    }

    #[test]
    fn pt_modulus_1_is_degenerate_but_doesnt_panic() {
        let params = make_params(&default_moduli(), 2, 1, 20, 6, 2, 1);
        assert_eq!(params.pt_modulus, 1);
    }

    // ----------------------------------------------------------------
    // Size computation methods
    // ----------------------------------------------------------------

    #[test]
    fn size_methods_are_nonzero_for_standard_params() {
        for params in [
            get_test_params(),
            get_short_keygen_params(),
            get_fast_expansion_testing_params(),
        ] {
            assert!(params.setup_bytes() > 0, "setup_bytes must be positive");
            assert!(params.query_bytes() > 0, "query_bytes must be positive");
            assert!(
                params.query_v_buf_bytes() > 0,
                "query_v_buf_bytes must be positive"
            );
            assert!(params.item_size() > 0, "item_size must be positive");
            assert!(
                params.bytes_per_chunk() > 0,
                "bytes_per_chunk must be positive"
            );
            assert!(
                params.modp_words_per_chunk() > 0,
                "modp_words_per_chunk must be positive"
            );
        }
    }

    // ----------------------------------------------------------------
    // Various n values
    // ----------------------------------------------------------------

    #[test]
    fn different_n_values_produce_valid_params() {
        for n in [1, 2, 3, 4, 5, 8] {
            let params = make_params(&default_moduli(), n, 256, 20, 6, 2, 1);
            assert_eq!(params.n, n);
            assert_eq!(params.num_items(), (1 << 6) * (1 << 2));
        }
    }

    // ----------------------------------------------------------------
    // Version field
    // ----------------------------------------------------------------

    #[test]
    fn version_0_and_nonzero_differ_in_setup_for_large_n() {
        // With n > 2, version 0 uses num_packing_mats = n, while version > 0 uses 2
        let p0 = Params::init(
            2048,
            &default_moduli(),
            6.4, 4, 256, 20, 4, 8, 56, 8, true, 9, 6, 1, 2048, 0,
        );
        let p1 = Params::init(
            2048,
            &default_moduli(),
            6.4, 4, 256, 20, 4, 8, 56, 8, true, 9, 6, 1, 2048, 1,
        );
        assert_ne!(
            p0.setup_bytes(),
            p1.setup_bytes(),
            "version 0 (n packing mats) vs version 1 (2 packing mats) should differ for n=4"
        );
    }

    #[test]
    fn version_field_is_stored() {
        for v in [0, 1, 2, 3] {
            let params = Params::init(
                2048,
                &default_moduli(),
                6.4, 2, 256, 20, 4, 8, 56, 8, true, 6, 2, 1, 2048, v,
            );
            assert_eq!(params.version, v);
        }
    }

    // ----------------------------------------------------------------
    // db_dim edge cases
    // ----------------------------------------------------------------

    #[test]
    fn db_dim_2_zero_produces_factor_on_first_dim_1() {
        let params = Params::init(
            2048,
            &default_moduli(),
            6.4, 2, 256, 20, 4, 8, 56, 8, true, 6, 0, 1, 2048, 0,
        );
        assert_eq!(params.db_dim_2, 0);
        assert_eq!(params.factor_on_first_dim(), 1);
        assert_eq!(params.num_items(), 1 << 6);
    }

    #[test]
    fn db_dim_2_nonzero_produces_factor_on_first_dim_2() {
        let params = make_params(&default_moduli(), 2, 256, 20, 6, 3, 1);
        assert_eq!(params.factor_on_first_dim(), 2);
        assert_eq!(params.num_items(), (1 << 6) * (1 << 3));
    }

    // ----------------------------------------------------------------
    // instances variations
    // ----------------------------------------------------------------

    #[test]
    fn instances_affects_item_size() {
        let p1 = make_params(&default_moduli(), 2, 256, 20, 6, 2, 1);
        let p4 = make_params(&default_moduli(), 2, 256, 20, 6, 2, 4);
        assert_eq!(p4.item_size(), 4 * p1.item_size());
    }

    // ----------------------------------------------------------------
    // Params equality / clone
    // ----------------------------------------------------------------

    #[test]
    fn params_clone_is_equal() {
        let params = get_test_params();
        let cloned = params.clone();
        assert_eq!(params, cloned);
    }

    #[test]
    fn different_params_are_not_equal() {
        let p1 = make_params(&default_moduli(), 2, 256, 20, 6, 2, 1);
        let p2 = make_params(&default_moduli(), 4, 256, 20, 6, 2, 1);
        assert_ne!(p1, p2);
    }
}
