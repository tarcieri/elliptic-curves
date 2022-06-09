//! Pure Rust implementation of group operations on secp256r1.

#[macro_use]
mod macros;

pub(crate) mod affine;
pub(crate) mod field;
#[cfg(feature = "hash2curve")]
mod hash2curve;
pub(crate) mod projective;
pub(crate) mod scalar;
pub(crate) mod util;

use affine::AffinePoint;
use field::{FieldElement, MODULUS};
use projective::ProjectivePoint;
use scalar::Scalar;

/// a = -3
const CURVE_EQUATION_A: FieldElement = FieldElement::ZERO
    .sub(&FieldElement::ONE)
    .sub(&FieldElement::ONE)
    .sub(&FieldElement::ONE);

/// b = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
const CURVE_EQUATION_B: FieldElement =
    FieldElement::from_be_hex("5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B");

#[cfg(test)]
mod tests {
    use super::{CURVE_EQUATION_A, CURVE_EQUATION_B};
    use hex_literal::hex;

    const CURVE_EQUATION_A_BYTES: &[u8] =
        &hex!("FFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFC");

    const CURVE_EQUATION_B_BYTES: &[u8] =
        &hex!("5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B");

    #[test]
    fn verify_constants() {
        assert_eq!(
            CURVE_EQUATION_A.to_bytes().as_slice(),
            CURVE_EQUATION_A_BYTES
        );
        assert_eq!(
            CURVE_EQUATION_B.to_bytes().as_slice(),
            CURVE_EQUATION_B_BYTES
        );
    }

    #[test]
    fn generate_secret_key() {
        use crate::SecretKey;
        use elliptic_curve::rand_core::OsRng;

        let key = SecretKey::random(&mut OsRng);

        // Sanity check
        assert!(!key.to_be_bytes().iter().all(|b| *b == 0))
    }
}
