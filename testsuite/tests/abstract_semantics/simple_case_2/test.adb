procedure Test is
   type My_Int is range -100000 .. 100000;
   subtype My_Nat is My_Int range 0 .. My_Int'Last;
   x : My_Int;
begin
   case x is
      when My_Nat =>
         x := 42;
      when others =>
         x := 31415;
   end case;
end Ex1;
