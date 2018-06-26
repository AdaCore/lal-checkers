procedure Ex1 is
   x : Integer;
   y : Integer;
   FOURTY_TWO : constant := 42;
begin
   case x is
      when FOURTY_TWO =>
         y := 43;
      when others =>
         y := x;
   end case;
end Ex1;
