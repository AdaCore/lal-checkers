procedure Test is
   X : Integer;

   procedure F is
   begin
      X := 12;
   end F;

   P : access procedure := F'Access;
begin
   P.all;
end Test;
