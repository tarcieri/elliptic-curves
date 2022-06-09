//! Field arithmetic modulo p = 2^{224}(2^{32} − 1) + 2^{192} + 2^{96} − 1

#![allow(clippy::assign_op_pattern, clippy::op_ref)]

#[cfg_attr(target_pointer_width = "32", path = "field/p256_32.rs")]
#[cfg_attr(target_pointer_width = "64", path = "field/p256_64.rs")]
mod field_impl;

use self::field_impl::*;
use crate::{FieldBytes, U256};
use core::ops::{AddAssign, Mul, MulAssign, Neg, SubAssign};
use elliptic_curve::{
    ff::Field,
    subtle::{Choice, ConstantTimeEq, CtOption},
};

/// Constant representing the modulus
/// p = 2^{224}(2^{32} − 1) + 2^{192} + 2^{96} − 1
pub const MODULUS: U256 =
    U256::from_be_hex("ffffffff00000001000000000000000000000000ffffffffffffffffffffffff");

/// An element in the finite field modulo p = 2^{224}(2^{32} − 1) + 2^{192} + 2^{96} − 1.
///
/// The internal representation is in little-endian order. Elements are always in
/// Montgomery form; i.e., FieldElement(a) = aR mod p, with R = 2^256.
#[derive(Clone, Copy, Debug)]
pub struct FieldElement(pub(crate) U256);

impl_field_element!(
    FieldElement,
    FieldBytes,
    U256,
    MODULUS,
    fiat_p256_montgomery_domain_field_element,
    fiat_p256_from_montgomery,
    fiat_p256_to_montgomery,
    fiat_p256_add,
    fiat_p256_sub,
    fiat_p256_mul,
    fiat_p256_opp,
    fiat_p256_square
);

impl FieldElement {
    /// Attempts to parse the given byte array as an SEC1-encoded field element.
    ///
    /// Returns None if the byte array does not contain a big-endian integer in the range
    /// [0, p).
    pub fn from_bytes(bytes: &FieldBytes) -> CtOption<Self> {
        Self::from_be_bytes(*bytes)
    }

    /// Returns the SEC1 encoding of this field element.
    pub fn to_bytes(self) -> FieldBytes {
        self.to_be_bytes()
    }

    /// Returns `self - rhs mod p`
    #[deprecated(since = "0.11.1", note = "please use `sub` instead")]
    pub const fn subtract(&self, rhs: &Self) -> Self {
        self.sub(rhs)
    }

    /// Returns `self^by`, where `by` is a little-endian integer exponent.
    ///
    /// **This operation is variable time with respect to the exponent.** If the exponent
    /// is fixed, this operation is effectively constant time.
    pub fn pow_vartime(&self, by: &[u64; 4]) -> Self {
        let mut res = Self::one();
        for e in by.iter().rev() {
            for i in (0..64).rev() {
                res = res.square();

                if ((*e >> i) & 1) == 1 {
                    res = res * self;
                }
            }
        }
        res
    }

    /// Returns the multiplicative inverse of self, if self is non-zero.
    pub fn invert(&self) -> CtOption<Self> {
        // We need to find b such that b * a ≡ 1 mod p. As we are in a prime
        // field, we can apply Fermat's Little Theorem:
        //
        //    a^p         ≡ a mod p
        //    a^(p-1)     ≡ 1 mod p
        //    a^(p-2) * a ≡ 1 mod p
        //
        // Thus inversion can be implemented with a single exponentiation.

        let t111 = self.mul(&self.mul(&self.square()).square());
        let t111111 = t111.mul(t111.sqn(3));
        let x15 = t111111.sqn(6).mul(t111111).sqn(3).mul(t111);
        let x16 = x15.square().mul(self);
        let i53 = x16.sqn(16).mul(x16).sqn(15);
        let x47 = x15.mul(i53);
        let inverse = x47
            .mul(i53.sqn(17).mul(self).sqn(143).mul(x47).sqn(47))
            .sqn(2)
            .mul(self);

        CtOption::new(inverse, !self.is_zero())
    }

    /// Returns the square root of self mod p, or `None` if no square root exists.
    pub fn sqrt(&self) -> CtOption<Self> {
        // We need to find alpha such that alpha^2 = beta mod p. For secp256r1,
        // p ≡ 3 mod 4. By Euler's Criterion, beta^(p-1)/2 ≡ 1 mod p. So:
        //
        //     alpha^2 = beta beta^((p - 1) / 2) mod p ≡ beta^((p + 1) / 2) mod p
        //     alpha = ± beta^((p + 1) / 4) mod p
        //
        // Thus sqrt can be implemented with a single exponentiation.

        let t11 = self.mul(&self.square());
        let t1111 = t11.mul(&t11.sqn(2));
        let t11111111 = t1111.mul(t1111.sqn(4));
        let x16 = t11111111.sqn(8).mul(t11111111);
        let sqrt = x16
            .sqn(16)
            .mul(x16)
            .sqn(32)
            .mul(self)
            .sqn(96)
            .mul(self)
            .sqn(94);

        CtOption::new(
            sqrt,
            (&sqrt * &sqrt).ct_eq(self), // Only return Some if it's the square root.
        )
    }

    /// Returns self^(2^n) mod p
    fn sqn(&self, n: usize) -> Self {
        let mut x = *self;
        for _ in 0..n {
            x = x.square();
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::FieldElement;
    use crate::{test_vectors::field::DBL_TEST_VECTORS, FieldBytes};
    use elliptic_curve::ff::Field;
    use proptest::{num::u64::ANY, prelude::*};

    #[test]
    fn zero_is_additive_identity() {
        let zero = FieldElement::zero();
        let one = FieldElement::one();
        assert_eq!(zero.add(&zero), zero);
        assert_eq!(one.add(&zero), one);
    }

    #[test]
    fn one_is_multiplicative_identity() {
        let one = FieldElement::one();
        assert_eq!(one.mul(&one), one);
    }

    #[test]
    fn from_bytes() {
        assert_eq!(
            FieldElement::from_bytes(&FieldBytes::default()).unwrap(),
            FieldElement::zero()
        );
        assert_eq!(
            FieldElement::from_bytes(
                &[
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 1
                ]
                .into()
            )
            .unwrap(),
            FieldElement::one()
        );
        assert!(bool::from(
            FieldElement::from_bytes(&[0xff; 32].into()).is_none()
        ));
    }

    #[test]
    fn to_bytes() {
        assert_eq!(FieldElement::zero().to_bytes(), FieldBytes::default());
        assert_eq!(
            FieldElement::one().to_bytes(),
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1
            ]
            .into()
        );
    }

    #[test]
    fn repeated_add() {
        let mut r = FieldElement::one();
        for i in 0..DBL_TEST_VECTORS.len() {
            assert_eq!(r.to_bytes(), DBL_TEST_VECTORS[i].into());
            r = r + &r;
        }
    }

    #[test]
    fn repeated_double() {
        let mut r = FieldElement::one();
        for i in 0..DBL_TEST_VECTORS.len() {
            assert_eq!(r.to_bytes(), DBL_TEST_VECTORS[i].into());
            r = r.double();
        }
    }

    #[test]
    fn repeated_mul() {
        let mut r = FieldElement::one();
        let two = r + &r;
        for i in 0..DBL_TEST_VECTORS.len() {
            assert_eq!(r.to_bytes(), DBL_TEST_VECTORS[i].into());
            r = r * &two;
        }
    }

    #[test]
    fn negation() {
        let two = FieldElement::one().double();
        let neg_two = -two;
        assert_eq!(two + &neg_two, FieldElement::zero());
        assert_eq!(-neg_two, two);
    }

    #[test]
    fn pow_vartime() {
        let one = FieldElement::one();
        let two = one + &one;
        let four = two.square();
        assert_eq!(two.pow_vartime(&[2, 0, 0, 0]), four);
    }

    #[test]
    fn invert() {
        assert!(bool::from(FieldElement::zero().invert().is_none()));

        let one = FieldElement::one();
        assert_eq!(one.invert().unwrap(), one);

        let two = one + &one;
        let inv_two = two.invert().unwrap();
        assert_eq!(two * &inv_two, one);
    }

    #[test]
    fn sqrt() {
        let one = FieldElement::one();
        let two = one + &one;
        let four = two.square();
        assert_eq!(four.sqrt().unwrap(), two);
    }

    #[cfg(target_pointer_width = "64")]
    proptest! {
        /// This checks behaviour well within the field ranges, because it doesn't set the
        /// highest limb.
        #[test]
        fn add_then_sub(
            a0 in ANY,
            a1 in ANY,
            a2 in ANY,
            b0 in ANY,
            b1 in ANY,
            b2 in ANY,
        ) {
            let a = FieldElement([a0, a1, a2, 0].into());
            let b = FieldElement([b0, b1, b2, 0].into());
            assert_eq!(a.add(&b).sub(&a), b);
        }
    }
}
