// Project UID 1d9f47bfc76643019cfbf037641defe1

#include "Card.h"
#include "unit_test_framework.h"
#include <iostream>

using namespace std;


TEST(test_card_ctor) {
    Card c(Card::RANK_ACE, Card::SUIT_HEARTS);
    ASSERT_EQUAL(Card::RANK_ACE, c.get_rank());
    ASSERT_EQUAL(Card::SUIT_HEARTS, c.get_suit());
}

TEST(test_card_ctor2) {
    Card c;
    ASSERT_EQUAL(Card::RANK_TWO, c.get_rank());
    ASSERT_EQUAL(Card::SUIT_SPADES, c.get_suit());
}

TEST(test_card_ctor3) {
    Card c(Card::RANK_KING, Card::SUIT_CLUBS);
    ASSERT_EQUAL(Card::RANK_KING, c.get_rank());
    ASSERT_EQUAL(Card::SUIT_CLUBS, c.get_suit());
}

TEST(test_card_ctor4) {
    Card c(Card::RANK_JACK, Card::SUIT_DIAMONDS);
    ASSERT_EQUAL(Card::RANK_JACK, c.get_rank());
    ASSERT_EQUAL(Card::SUIT_DIAMONDS, c.get_suit());
}

TEST(test_card_ctor5) {
    Card c(Card::RANK_ACE, Card::SUIT_HEARTS);
    ASSERT_EQUAL(Card::RANK_ACE, c.get_rank());
    ASSERT_EQUAL(Card::SUIT_HEARTS, c.get_suit());
}

TEST(test_card_ctor6) {
    Card c(Card::RANK_FOUR, Card::SUIT_HEARTS);
    ASSERT_EQUAL(Card::RANK_FOUR, c.get_rank());
    ASSERT_EQUAL(Card::SUIT_HEARTS, c.get_suit());
}

TEST(test_card_ctor7) {
    Card c(Card::RANK_ACE, Card::SUIT_HEARTS);
    ASSERT_EQUAL(Card::RANK_ACE, c.get_rank());
    ASSERT_EQUAL(Card::SUIT_HEARTS, c.get_suit());
}

TEST(test_card_ctor8) {
    Card c(Card::RANK_FOUR, Card::SUIT_HEARTS);
    ASSERT_EQUAL(Card::RANK_FOUR, c.get_rank());
    ASSERT_EQUAL(Card::SUIT_HEARTS, c.get_suit(Card::SUIT_DIAMONDS));
}

TEST(test_card_ctor9) {
    Card c(Card::RANK_TEN, Card::SUIT_DIAMONDS);
    ASSERT_EQUAL(Card::RANK_TEN, c.get_rank());
    ASSERT_EQUAL(Card::SUIT_DIAMONDS, c.get_suit(Card::SUIT_DIAMONDS));
}

TEST(test_card_get_suit_special1) {
    Card c(Card::RANK_JACK, Card::SUIT_DIAMONDS);
    ASSERT_EQUAL(Card::RANK_JACK, c.get_rank());
    ASSERT_EQUAL(Card::SUIT_HEARTS, c.get_suit(Card::SUIT_HEARTS));
}

TEST(test_card_get_suit_special2) {
    Card c(Card::RANK_JACK, Card::SUIT_HEARTS);
    ASSERT_EQUAL(Card::RANK_JACK, c.get_rank());
    ASSERT_EQUAL(Card::SUIT_DIAMONDS, c.get_suit(Card::SUIT_DIAMONDS));
}


TEST(test_card_get_suit_special3) {
    Card c(Card::RANK_JACK, Card::SUIT_CLUBS);
    ASSERT_EQUAL(Card::RANK_JACK, c.get_rank());
    ASSERT_EQUAL(Card::SUIT_SPADES, c.get_suit(Card::SUIT_SPADES));
}

TEST(test_card_get_suit_special4) {
    Card c(Card::RANK_JACK, Card::SUIT_SPADES);
    ASSERT_EQUAL(Card::RANK_JACK, c.get_rank());
    ASSERT_EQUAL(Card::SUIT_CLUBS, c.get_suit(Card::SUIT_CLUBS));
}

TEST(test_card_suit_and_rank) {
    Card two_spades = Card();
    ASSERT_EQUAL(two_spades.get_rank(), Card::RANK_TWO);
    ASSERT_EQUAL(two_spades.get_suit(), Card::SUIT_SPADES);

    Card three_spades = Card(Card::RANK_THREE, Card::SUIT_SPADES);
    ASSERT_EQUAL(three_spades.get_rank(), Card::RANK_THREE);
    ASSERT_EQUAL(three_spades.get_suit(), Card::SUIT_SPADES);
    ASSERT_EQUAL(three_spades.get_suit(Card::SUIT_CLUBS), Card::SUIT_SPADES);
}

TEST(test_card_type) {
    Card three_spades = Card(Card::RANK_THREE, Card::SUIT_SPADES);
    ASSERT_FALSE(three_spades.is_face());
    ASSERT_FALSE(three_spades.is_right_bower(Card::SUIT_CLUBS));
    ASSERT_FALSE(three_spades.is_left_bower(Card::SUIT_CLUBS));
    ASSERT_FALSE(three_spades.is_trump(Card::SUIT_CLUBS));
}

TEST(test_card_type1) {
    Card ten_diamonds = Card(Card::RANK_TEN, Card::SUIT_DIAMONDS);
    ASSERT_FALSE(ten_diamonds.is_face());
    ASSERT_FALSE(ten_diamonds.is_right_bower(Card::SUIT_DIAMONDS));
    ASSERT_FALSE(ten_diamonds.is_left_bower(Card::SUIT_DIAMONDS));
    ASSERT_FALSE(ten_diamonds.is_trump(Card::SUIT_CLUBS));
    ASSERT_TRUE(ten_diamonds.is_trump(Card::SUIT_DIAMONDS));
}

TEST(test_card_type2) {
    Card c = Card(Card::RANK_JACK, Card::SUIT_DIAMONDS);
    ASSERT_TRUE(c.is_face());
    ASSERT_TRUE(c.is_right_bower(Card::SUIT_DIAMONDS));
    ASSERT_FALSE(c.is_right_bower(Card::SUIT_CLUBS));
    ASSERT_FALSE(c.is_right_bower(Card::SUIT_HEARTS));
    ASSERT_FALSE(c.is_left_bower(Card::SUIT_DIAMONDS));
    ASSERT_FALSE(c.is_left_bower(Card::SUIT_SPADES));
    ASSERT_TRUE(c.is_left_bower(Card::SUIT_HEARTS));
    ASSERT_FALSE(c.is_trump(Card::SUIT_CLUBS));
    ASSERT_TRUE(c.is_trump(Card::SUIT_DIAMONDS));
}

TEST(test_card_type3) {
    Card c = Card(Card::RANK_KING, Card::SUIT_DIAMONDS);
    ASSERT_TRUE(c.is_face());
    ASSERT_FALSE(c.is_right_bower(Card::SUIT_DIAMONDS));
    ASSERT_FALSE(c.is_right_bower(Card::SUIT_CLUBS));
    ASSERT_FALSE(c.is_right_bower(Card::SUIT_HEARTS));
    ASSERT_FALSE(c.is_left_bower(Card::SUIT_DIAMONDS));
    ASSERT_FALSE(c.is_left_bower(Card::SUIT_SPADES));
    ASSERT_FALSE(c.is_left_bower(Card::SUIT_HEARTS));
    ASSERT_FALSE(c.is_trump(Card::SUIT_CLUBS));
    ASSERT_TRUE(c.is_trump(Card::SUIT_DIAMONDS));
}

TEST(test_card_self_comparison) {
    Card three_spades = Card(Card::RANK_THREE, Card::SUIT_SPADES);
    ASSERT_FALSE(three_spades < three_spades);
    ASSERT_TRUE(three_spades <= three_spades);
    ASSERT_FALSE(three_spades > three_spades);
    ASSERT_TRUE(three_spades >= three_spades);
    ASSERT_TRUE(three_spades == three_spades);
    ASSERT_FALSE(three_spades != three_spades);
}

TEST(test_Suit_next) {
    ASSERT_EQUAL(Suit_next(Card::SUIT_CLUBS), Card::SUIT_SPADES);
}

TEST(test_Suit_next1) {
    ASSERT_EQUAL(Suit_next(Card::SUIT_SPADES), Card::SUIT_CLUBS);
}

TEST(test_Suit_next2) {
    ASSERT_EQUAL(Suit_next(Card::SUIT_HEARTS), Card::SUIT_DIAMONDS);
}

TEST(test_Suit_next3) {
    ASSERT_EQUAL(Suit_next(Card::SUIT_DIAMONDS), Card::SUIT_HEARTS);
}

TEST(test_Card_less_self) {
    Card three_spades = Card(Card::RANK_THREE, Card::SUIT_SPADES);
    ASSERT_FALSE(Card_less(three_spades, three_spades, Card::SUIT_CLUBS));
    ASSERT_FALSE(Card_less(three_spades, three_spades, three_spades,
                           Card::SUIT_CLUBS));
}

TEST(test_card_insertion) {
    Card three_spades = Card(Card::RANK_THREE, Card::SUIT_SPADES);
    ostringstream oss;
    oss << three_spades;
    ASSERT_EQUAL(oss.str(), "Three of Spades");
}

TEST(test_card_insertion1) {
    Card ace_diamonds = Card(Card::RANK_ACE, Card::SUIT_DIAMONDS);
    ostringstream oss;
    oss << ace_diamonds;
    ASSERT_EQUAL(oss.str(), "Ace of Diamonds");
}

TEST(test_card_insertion2) {
    Card jack_hearts = Card(Card::RANK_JACK, Card::SUIT_HEARTS);
    ostringstream oss;
    oss << jack_hearts;
    ASSERT_EQUAL(oss.str(), "Jack of Hearts");
}

TEST(test_operator) {
    Card nine_spades(Card::RANK_NINE, Card::SUIT_SPADES);
    Card ten_spades(Card::RANK_TEN, Card::SUIT_SPADES);
    Card nine_hearts(Card::RANK_NINE, Card::SUIT_HEARTS);
    Card nine_clubs(Card::RANK_NINE, Card::SUIT_CLUBS);
    Card nine_diamonds(Card::RANK_NINE, Card::SUIT_DIAMONDS);
    Card jack_hearts(Card::RANK_JACK, Card::SUIT_HEARTS);
    Card ace_spades(Card::RANK_ACE, Card::SUIT_SPADES);
    Card king_clubs(Card::RANK_KING, Card::SUIT_CLUBS);
    Card ace_diamonds(Card::RANK_ACE, Card::SUIT_DIAMONDS);
    Card ten_diamonds(Card::RANK_TEN, Card::SUIT_DIAMONDS);


    ASSERT_TRUE(nine_spades < ten_spades);
    ASSERT_TRUE(nine_spades < nine_hearts);
    ASSERT_TRUE(nine_hearts < ten_spades);
    ASSERT_FALSE(nine_spades > nine_spades);
    ASSERT_FALSE(nine_spades > nine_hearts);
    ASSERT_TRUE(nine_spades >= nine_spades);
    ASSERT_TRUE(nine_spades <= nine_spades);
    ASSERT_TRUE(nine_hearts >= nine_spades);
    ASSERT_TRUE(nine_spades == nine_spades);
    ASSERT_FALSE(nine_spades != nine_spades);
    ASSERT_TRUE(ten_spades != nine_spades);
    ASSERT_TRUE(nine_spades != nine_hearts);
    ASSERT_TRUE(ten_spades >= nine_spades);
    ASSERT_TRUE(nine_diamonds <= ace_diamonds);
    ASSERT_TRUE(nine_diamonds < ace_diamonds);
    ASSERT_TRUE(nine_clubs < ten_spades);
    ASSERT_TRUE(ace_spades > ten_diamonds);
    ASSERT_TRUE(ace_spades > jack_hearts);
    ASSERT_TRUE(ace_spades <= ace_diamonds);
    ASSERT_TRUE(king_clubs >= ten_diamonds);
    ASSERT_TRUE(king_clubs != ten_diamonds);
    ASSERT_FALSE(king_clubs == ten_diamonds);
    ASSERT_TRUE(ace_diamonds >= jack_hearts);
    ASSERT_TRUE(ace_diamonds > king_clubs);
    ASSERT_TRUE(king_clubs > jack_hearts);
    ASSERT_TRUE(king_clubs >= nine_clubs);
    ASSERT_TRUE(king_clubs > ten_spades);
    ASSERT_TRUE(ten_spades <= jack_hearts);

}

TEST(test_card_less_wo_lead) {
    Card nine_spades(Card::RANK_NINE, Card::SUIT_SPADES);
    Card nine_hearts(Card::RANK_NINE, Card::SUIT_HEARTS);
    Card ace_diamonds(Card::RANK_ACE, Card::SUIT_DIAMONDS);
    Card ace_clubs(Card::RANK_ACE, Card::SUIT_CLUBS);
    Card ace_spades(Card::RANK_ACE, Card::SUIT_SPADES);
    Card jack_spades(Card::RANK_JACK, Card::SUIT_SPADES);
    Card jack_clubs(Card::RANK_JACK, Card::SUIT_CLUBS);
    Card jack_diamonds(Card::RANK_JACK, Card::SUIT_DIAMONDS);

    std::string trump = "Spades";

    ASSERT_TRUE(Card_less(ace_diamonds, nine_spades, trump));
    ASSERT_TRUE(Card_less(nine_hearts, nine_spades, trump));
    ASSERT_FALSE(Card_less(nine_spades, nine_spades, trump));
    ASSERT_TRUE(Card_less(jack_clubs, jack_spades, trump));
    ASSERT_TRUE(Card_less(jack_diamonds, ace_clubs, trump));
    ASSERT_TRUE(Card_less(jack_diamonds, ace_spades, trump));
    ASSERT_FALSE(Card_less(jack_spades, jack_spades, trump));
    ASSERT_TRUE(Card_less(ace_clubs, jack_clubs, trump));
    ASSERT_TRUE(Card_less(nine_spades, jack_clubs, trump));
}

TEST(test_card_less_wo_lead2) {
    Card nine_spades(Card::RANK_NINE, Card::SUIT_SPADES);
    Card nine_hearts(Card::RANK_NINE, Card::SUIT_HEARTS);
    Card nine_diamonds(Card::RANK_NINE, Card::SUIT_DIAMONDS);
    Card king_spades(Card::RANK_KING, Card::SUIT_SPADES);
    Card king_clubs(Card::RANK_KING, Card::SUIT_CLUBS);
    Card ace_diamonds(Card::RANK_ACE, Card::SUIT_DIAMONDS);
    Card ace_hearts(Card::RANK_ACE, Card::SUIT_HEARTS);
    Card jack_spades(Card::RANK_JACK, Card::SUIT_SPADES);
    Card jack_hearts(Card::RANK_JACK, Card::SUIT_HEARTS);
    Card jack_diamonds(Card::RANK_JACK, Card::SUIT_DIAMONDS);
    Card jack_clubs(Card::RANK_JACK, Card::SUIT_CLUBS);

    std::string trump = "Hearts";
    ASSERT_TRUE(Card_less(nine_diamonds, nine_hearts, trump));
    ASSERT_TRUE(Card_less(ace_diamonds, nine_hearts, trump));
    ASSERT_TRUE(Card_less(nine_hearts, jack_diamonds, trump));
    ASSERT_TRUE(Card_less(ace_diamonds, jack_diamonds, trump));
    ASSERT_TRUE(Card_less(jack_diamonds, jack_hearts, trump));
    ASSERT_TRUE(Card_less(ace_hearts, jack_diamonds, trump));
    ASSERT_TRUE(Card_less(jack_spades, king_spades, trump));
    ASSERT_TRUE(Card_less(jack_clubs, king_spades, trump));
}

TEST(test_card_less_w_lead) {
    Card nine_spades(Card::RANK_NINE, Card::SUIT_SPADES);
    Card nine_hearts(Card::RANK_NINE, Card::SUIT_HEARTS);
    Card ace_diamonds(Card::RANK_ACE, Card::SUIT_DIAMONDS);
    Card ace_spades(Card::RANK_ACE, Card::SUIT_SPADES);
    Card ten_clubs(Card::RANK_TEN, Card::SUIT_CLUBS);
    Card ten_diamonds(Card::RANK_TEN, Card::SUIT_DIAMONDS);
    Card queen_hearts(Card::RANK_QUEEN, Card::SUIT_HEARTS);
    Card king_spades(Card::RANK_KING, Card::SUIT_SPADES);
    Card king_clubs(Card::RANK_KING, Card::SUIT_CLUBS);
    Card king_hearts(Card::RANK_KING, Card::SUIT_HEARTS);
    Card jack_Spades(Card::RANK_JACK, Card::SUIT_SPADES);
    Card jack_clubs(Card::RANK_JACK, Card::SUIT_CLUBS);
    Card jack_hearts(Card::RANK_JACK, Card::SUIT_HEARTS);
    Card jack_diamonds(Card::RANK_JACK, Card::SUIT_DIAMONDS);
    
    Card led_card(Card::RANK_NINE, Card::SUIT_HEARTS);

    std::string trump = "Spades";

    ASSERT_FALSE(Card_less(nine_spades, ace_diamonds, led_card, trump));
    ASSERT_FALSE(Card_less(nine_hearts, ace_diamonds, led_card, trump));
    ASSERT_FALSE(Card_less(nine_spades, nine_hearts, led_card, trump));
    ASSERT_TRUE(Card_less(jack_clubs, jack_Spades, led_card, trump));
    ASSERT_TRUE(Card_less(jack_hearts, jack_clubs, led_card, trump));
    ASSERT_TRUE(Card_less(jack_hearts, jack_Spades, led_card, trump));
    ASSERT_TRUE(Card_less(jack_diamonds, jack_hearts, led_card, trump));
    ASSERT_TRUE(Card_less(jack_diamonds, jack_clubs, led_card, trump));
    ASSERT_TRUE(Card_less(jack_diamonds, jack_Spades, led_card, trump));
    ASSERT_TRUE(Card_less(ace_diamonds, jack_clubs, led_card, trump));
    ASSERT_TRUE(Card_less(king_spades, jack_Spades, led_card, trump));
    ASSERT_TRUE(Card_less(king_clubs, king_spades, led_card, trump));
    ASSERT_TRUE(Card_less(king_hearts, king_spades, led_card, trump));
    ASSERT_TRUE(Card_less(king_clubs, king_hearts, led_card, trump));
}

TEST_MAIN()
