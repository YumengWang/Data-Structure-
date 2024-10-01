// Project UID 1d9f47bfc76643019cfbf037641defe1

#include "Pack.h"
#include "unit_test_framework.h"

#include <iostream>

using namespace std;

static const int PACK_SIZE = 24;

TEST(test_pack_default_ctor) {
    Pack pack;
    Card first = pack.deal_one();
    ASSERT_EQUAL(Card::RANK_NINE, first.get_rank());
    ASSERT_EQUAL(Card::SUIT_SPADES, first.get_suit());
}

/*TEST(test_pack_istream_ctor) {
    const string filename = "pack.in";
    ifstream ifs(filename);
    assert(ifs.is_open());
    Pack pack(ifs);
    Card first_card = pack.deal_one();
    ASSERT_EQUAL(Card::RANK_NINE, first_card.get_rank());
    ASSERT_EQUAL(Card::SUIT_SPADES, first_card.get_suit());
}*/

TEST(test_pack_reset) {
  Pack pack;
  pack.deal_one();
  pack.reset();
  Card first_card = pack.deal_one();
  ASSERT_EQUAL(first_card, Card(Card::RANK_NINE, Card::SUIT_SPADES));
}

TEST(test_pack_empty) {
  Pack pack;
  for (int i = 0; i < PACK_SIZE - 1; ++i) {
    pack.deal_one();
    ASSERT_FALSE(pack.empty());
  }
  pack.deal_one();
  ASSERT_TRUE(pack.empty());
}

TEST(test_pack_shuffle) {
    Pack pack;
    pack.shuffle();
    Card first_card = pack.deal_one();
    ASSERT_EQUAL(Card::RANK_KING, first_card.get_rank());
    ASSERT_EQUAL(Card::SUIT_CLUBS, first_card.get_suit());
}

TEST(test_pack_shuffle2) {
    Pack pack;
    pack.shuffle();
    Card two_card = pack.deal_one();
    two_card = pack.deal_one();
    ASSERT_EQUAL(Card::RANK_JACK, two_card.get_rank());
    ASSERT_EQUAL(Card::SUIT_HEARTS, two_card.get_suit());
}

TEST(test_pack_shuffle3) {
    Pack pack;
    pack.shuffle();
    Card three_card = pack.deal_one();
    three_card = pack.deal_one();
    three_card = pack.deal_one();
    ASSERT_EQUAL(Card::RANK_NINE, three_card.get_rank());
    ASSERT_EQUAL(Card::SUIT_SPADES, three_card.get_suit());
}

TEST(test_pack_shuffle4) {
    Pack pack;
    pack.shuffle();
    Card four_card = pack.deal_one();
    four_card = pack.deal_one();
    four_card = pack.deal_one();
    four_card = pack.deal_one();
    ASSERT_EQUAL(Card::RANK_ACE, four_card.get_rank());
    ASSERT_EQUAL(Card::SUIT_CLUBS, four_card.get_suit());
}

TEST(test_pack_shuffle5) {
    Pack pack;
    pack.shuffle();
    pack.shuffle();
    Card three_card = pack.deal_one();
    three_card = pack.deal_one();
    three_card = pack.deal_one();
    ASSERT_EQUAL(Card::RANK_KING, three_card.get_rank());
    ASSERT_EQUAL(Card::SUIT_CLUBS, three_card.get_suit());
}

TEST(test_deal_card) {
    Pack pack;
    pack.deal_one();
    pack.deal_one();
    Card third_card = pack.deal_one();
    ASSERT_EQUAL(Card::RANK_JACK, third_card.get_rank());
    ASSERT_EQUAL(Card::SUIT_SPADES, third_card.get_suit());
}

TEST_MAIN()
