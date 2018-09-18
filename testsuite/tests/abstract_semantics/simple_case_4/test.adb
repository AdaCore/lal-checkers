procedure Test is
   subtype A_B_OR_C is Character range 'a' .. 'c';
   C : Character;
begin
   case C is
      when A_B_OR_C =>
         C := 'y';
      when others =>
         C := 'z';
   end case;
end Test;
