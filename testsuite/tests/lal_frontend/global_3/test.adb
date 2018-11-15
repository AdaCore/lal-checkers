procedure Test is
   procedure F (X : Integer);

   procedure G is
   begin
      F (X => 2);  --  X should not be considered global
   end G;

   procedure F (X : Integer) is
   begin
      null;
   end F;
begin
   G;
end Test;
