//! secp384r1 scalar field elements.

#![allow(clippy::unusual_byte_groupings)]

#[cfg_attr(target_pointer_width = "32", path = "scalar/p384_scalar_32.rs")]
#[cfg_attr(target_pointer_width = "64", path = "scalar/p384_scalar_64.rs")]
#[allow(
    clippy::identity_op,
    clippy::too_many_arguments,
    clippy::unnecessary_cast
)]
mod scalar_impl;

use self::scalar_impl::*;
use crate::{FieldBytes, NistP384, SecretKey, U384};
use core::ops::{AddAssign, MulAssign, Neg, SubAssign};
use elliptic_curve::{
    bigint::{self, ArrayEncoding, Encoding, Integer, Limb},
    ff::PrimeField,
    ops::Reduce,
    subtle::{
        Choice, ConditionallySelectable, ConstantTimeEq, ConstantTimeGreater, ConstantTimeLess,
        CtOption,
    },
    Curve as _, Error, IsHigh, Result, ScalarArithmetic, ScalarCore,
};

#[cfg(feature = "bits")]
use {crate::ScalarBits, elliptic_curve::group::ff::PrimeFieldBits};

impl ScalarArithmetic for NistP384 {
    type Scalar = Scalar;
}

/// Scalars are elements in the finite field modulo n.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(docsrs, doc(cfg(feature = "arithmetic")))]
pub struct Scalar(U384);

impl_field_element!(
    Scalar,
    FieldBytes,
    U384,
    NistP384::ORDER,
    fiat_p384_scalar_montgomery_domain_field_element,
    fiat_p384_scalar_from_montgomery,
    fiat_p384_scalar_to_montgomery,
    fiat_p384_scalar_add,
    fiat_p384_scalar_sub,
    fiat_p384_scalar_mul,
    fiat_p384_scalar_opp,
    fiat_p384_scalar_square
);

impl Scalar {
    /// `2^s` root of unity.
    pub const ROOT_OF_UNITY: Self = Self::from_be_hex("ffffffffffffffffffffffffffffffffffffffffffffffffc7634d81f4372ddf581a0db248b0a77aecec196accc52972");

    /// Compute [`Scalar`] inversion: `1 / self`.
    pub fn invert(&self) -> CtOption<Self> {
        let ret = impl_field_invert!(
            self.to_canonical().to_uint_array(),
            Self::ONE.0.to_uint_array(),
            Limb::BIT_SIZE,
            bigint::nlimbs!(U384::BIT_SIZE),
            fiat_p384_scalar_mul,
            fiat_p384_scalar_opp,
            fiat_p384_scalar_divstep_precomp,
            fiat_p384_scalar_divstep,
            fiat_p384_scalar_msat,
            fiat_p384_scalar_selectznz,
        );
        CtOption::new(Self(ret.into()), !self.is_zero())
    }

    /// Compute modular square root.
    pub fn sqrt(&self) -> CtOption<Self> {
        // p mod 4 = 3 -> compute sqrt(x) using x^((p+1)/4) =
        // x^9850501549098619803069760025035903451269934817616361666986726319906914849778315892349739077038073728388608413485661
        let t1 = *self;
        let t10 = t1.square();
        let t11 = *self * t10;
        let t101 = t10 * t11;
        let t111 = t10 * t101;
        let t1001 = t10 * t111;
        let t1011 = t10 * t1001;
        let t1101 = t10 * t1011;
        let t1111 = t10 * t1101;
        let t11110 = t1111.square();
        let t11111 = t1 * t11110;
        let t1111100 = t11111.sqn(2);
        let t11111000 = t1111100.square();
        let i14 = t11111000.square();
        let i20 = i14.sqn(5) * i14;
        let i31 = i20.sqn(10) * i20;
        let i58 = (i31.sqn(4) * t11111000).sqn(21) * i31;
        let i110 = (i58.sqn(3) * t1111100).sqn(47) * i58;
        let x194 = i110.sqn(95) * i110 * t1111;
        let i225 = ((x194.sqn(6) * t111).sqn(3) * t11).sqn(7);
        let i235 = ((t1101 * i225).sqn(6) * t1101).square() * t1;
        let i258 = ((i235.sqn(11) * t11111).sqn(2) * t1).sqn(8);
        let i269 = ((t1101 * i258).sqn(2) * t11).sqn(6) * t1011;
        let i286 = ((i269.sqn(4) * t111).sqn(6) * t11111).sqn(5);
        let i308 = ((t1011 * i286).sqn(10) * t1101).sqn(9) * t1101;
        let i323 = ((i308.sqn(4) * t1011).sqn(6) * t1001).sqn(3);
        let i340 = ((t1 * i323).sqn(7) * t1011).sqn(7) * t101;
        let i357 = ((i340.sqn(5) * t111).sqn(5) * t1111).sqn(5);
        let i369 = ((t1011 * i357).sqn(4) * t1011).sqn(5) * t111;
        let i387 = ((i369.sqn(3) * t11).sqn(7) * t11).sqn(6);
        let i397 = ((t1011 * i387).sqn(4) * t101).sqn(3) * t11;
        let i413 = ((i397.sqn(4) * t11).sqn(4) * t11).sqn(6);
        let i427 = ((t101 * i413).sqn(5) * t101).sqn(6) * t1011;
        let x = i427.sqn(3) * t101;
        CtOption::new(x, x.square().ct_eq(&t1))
    }

    fn sqn(&self, n: usize) -> Self {
        let mut x = *self;
        for _ in 0..n {
            x = x.square();
        }
        x
    }

    /// Returns the SEC1 encoding of this scalar.
    ///
    /// Required for running test vectors.
    #[cfg(test)]
    pub fn to_bytes(&self) -> FieldBytes {
        self.to_be_bytes()
    }
}

impl IsHigh for Scalar {
    fn is_high(&self) -> Choice {
        const MODULUS_SHR1: U384 = NistP384::ORDER.shr_vartime(1);
        self.to_canonical().ct_gt(&MODULUS_SHR1)
    }
}

impl PrimeField for Scalar {
    type Repr = FieldBytes;

    const CAPACITY: u32 = 383;
    const NUM_BITS: u32 = 384;
    const S: u32 = 1;

    fn from_repr(bytes: FieldBytes) -> CtOption<Self> {
        Self::from_be_bytes(bytes)
    }

    fn to_repr(&self) -> FieldBytes {
        self.to_be_bytes()
    }

    fn is_odd(&self) -> Choice {
        self.is_odd()
    }

    fn multiplicative_generator() -> Self {
        2u64.into()
    }

    fn root_of_unity() -> Self {
        Self::ROOT_OF_UNITY
    }
}

#[cfg(feature = "bits")]
#[cfg_attr(docsrs, doc(cfg(feature = "bits")))]
impl PrimeFieldBits for Scalar {
    type ReprBits = fiat_p384_scalar_montgomery_domain_field_element;

    fn to_le_bits(&self) -> ScalarBits {
        self.to_canonical().to_uint_array().into()
    }

    fn char_le_bits() -> ScalarBits {
        NistP384::ORDER.to_uint_array().into()
    }
}

impl Reduce<U384> for Scalar {
    fn from_uint_reduced(w: U384) -> Self {
        let (r, underflow) = w.sbb(&NistP384::ORDER, Limb::ZERO);
        let underflow = Choice::from((underflow.0 >> (Limb::BIT_SIZE - 1)) as u8);
        Self::from_uint_unchecked(U384::conditional_select(&w, &r, !underflow))
    }
}

impl From<u64> for Scalar {
    fn from(n: u64) -> Scalar {
        Self::from_uint_unchecked(U384::from(n))
    }
}

impl From<ScalarCore<NistP384>> for Scalar {
    fn from(w: ScalarCore<NistP384>) -> Self {
        Scalar::from(&w)
    }
}

impl From<&ScalarCore<NistP384>> for Scalar {
    fn from(w: &ScalarCore<NistP384>) -> Scalar {
        Scalar::from_uint_unchecked(*w.as_uint())
    }
}

impl From<Scalar> for ScalarCore<NistP384> {
    fn from(scalar: Scalar) -> ScalarCore<NistP384> {
        ScalarCore::from(&scalar)
    }
}

impl From<&Scalar> for ScalarCore<NistP384> {
    fn from(scalar: &Scalar) -> ScalarCore<NistP384> {
        ScalarCore::new(scalar.into()).unwrap()
    }
}

impl From<Scalar> for FieldBytes {
    fn from(scalar: Scalar) -> Self {
        scalar.to_repr()
    }
}

impl From<&Scalar> for FieldBytes {
    fn from(scalar: &Scalar) -> Self {
        scalar.to_repr()
    }
}

impl From<Scalar> for U384 {
    fn from(scalar: Scalar) -> U384 {
        U384::from(&scalar)
    }
}

impl From<&Scalar> for U384 {
    fn from(scalar: &Scalar) -> U384 {
        scalar.to_canonical()
    }
}

impl From<&SecretKey> for Scalar {
    fn from(secret_key: &SecretKey) -> Scalar {
        *secret_key.to_nonzero_scalar()
    }
}

impl TryFrom<U384> for Scalar {
    type Error = Error;

    fn try_from(w: U384) -> Result<Self> {
        Option::from(Self::from_uint(w)).ok_or(Error)
    }
}

#[cfg(test)]
mod tests {
    use super::Scalar;
    use crate::FieldBytes;
    use elliptic_curve::ff::{Field, PrimeField};

    #[test]
    fn from_to_bytes_roundtrip() {
        let k: u64 = 42;
        let mut bytes = FieldBytes::default();
        bytes[40..].copy_from_slice(k.to_le_bytes().as_ref());

        let scalar = Scalar::from_repr(bytes).unwrap();
        assert_eq!(bytes, scalar.to_be_bytes());
    }

    /// Basic tests that multiplication works.
    #[test]
    fn multiply() {
        let one = Scalar::one();
        let two = one + one;
        let three = two + one;
        let six = three + three;
        assert_eq!(six, two * three);

        let minus_two = -two;
        let minus_three = -three;
        assert_eq!(two, -minus_two);

        assert_eq!(minus_three * minus_two, minus_two * minus_three);
        assert_eq!(six, minus_two * minus_three);
    }

    /// Basic tests that scalar inversion works.
    #[test]
    fn invert() {
        let one = Scalar::one();
        let three = one + one + one;
        let inv_three = three.invert().unwrap();
        assert_eq!(three * inv_three, one);

        let minus_three = -three;
        let inv_minus_three = minus_three.invert().unwrap();
        assert_eq!(inv_minus_three, -inv_three);
        assert_eq!(three * inv_minus_three, -one);
    }

    /// Basic tests that sqrt works.
    #[test]
    fn sqrt() {
        for &n in &[1u64, 4, 9, 16, 25, 36, 49, 64] {
            let scalar = Scalar::from(n);
            let sqrt = scalar.sqrt().unwrap();
            assert_eq!(sqrt.square(), scalar);
        }
    }
}
